import argparse
from functools import reduce
import os
import os.path as path
import numpy as np
import pickle
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csc_matrix
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra

from build_reference_map import read_descriptors, build_map, load_subsampled_data
from archived.hmm_inference import viterbi, forward_algorithm, online_localization
from measurement_model import vmflhood
from motion_model import create_transition_matrix, create_deviation_matrix
from settings import DATA_DIR
import geometry


def plot_viterbi(pred, poses, posesQ, sims):
    # extract gt allocation of query poses
    gt = np.asarray([np.argmin(geometry.metric(Q, poses, w)) for Q in posesQ])
    # setup plots
    fig, ax = plt.subplots(1, 2)
    # plot batch state estimate from viterbi
    # true query pose (red) matched to best refn match
    T = len(pred)
    xyzrpy = poses.to_xyzrpy()
    xyzrpyQ = posesQ.to_xyzrpy()
    ax[0].set_title('Sequence localizer (Viterbi)')
    ax[0].scatter(xyzrpy[:, 1], xyzrpy[:, 0], color='black', s=5)
    ax[0].scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='green', s=5)
    ax[0].axis('square')
    for t in range(T):
        px = np.vstack((xyzrpyQ[t, 1], xyzrpy[gt[t], 1]))
        py = np.vstack((xyzrpyQ[t, 0], xyzrpy[gt[t], 0]))
        ax[0].plot(px, py, 'g-')
    # plot best match based on single image retrieval only

    # extract image retrieval indices
    IR_inds = np.argmax(sims, axis=1)
    # setup figure
    ax[1].set_title('Single image retrieval')
    ax[1].scatter(xyzrpy[:, 1], xyzrpy[:, 0], color='black', s=5)
    ax[1].scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='red', s=5)
    ax[1].axis('square')
    for t in range(T):
        px = np.vstack((xyzrpyQ[t, 1], xyzrpy[IR_inds[t], 1]))
        py = np.vstack((xyzrpyQ[t, 0], xyzrpy[IR_inds[t], 0]))
        ax[1].plot(px, py, 'r-')
    plt.show()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverse", type=str, default="night",
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str, default='t_1_w_5.csv',
                    help="filename containing subsampled reference traverse poses")
    parser.add_argument("-qf", "--query-filename", type=str, default='t_1_w_5.csv',
                    help="filename containing subsampled query traverse poses")
    parser.add_argument("-w", "--attitude-weight", type=float, default=5,
        help=("weight for attitude component of pose distances equal to d where"
              "1 / d being rotation angle (rad) equivalent to 1m translation"))
    parser.add_argument("-p", "--pca-dim", type=int, default=4096,
                        help="number of dimensions for nv descriptor")
    parser.add_argument("-s", "--start", type=int, default=0, help="start index of query sequence")
    parser.add_argument("-L", "--length", type=int, default=30, help="query sequence length for viterbi")
    args = parser.parse_args()

    # configuration variables

    w = args.attitude_weight;
    ref_traverse = args.reference_traverse
    query_traverse = args.query_traverse
    ref_fname = args.reference_filename
    q_fname = args.query_filename
    pca_dim = args.pca_dim
    start_ind = args.start
    end_ind = start_ind + args.length
    T = args.length

    att_wt = np.ones(6)
    att_wt[3:] *= w

    # parameters
    Eoo = 0.7
    theta = np.ones((T - 1, 3))
    theta[:, 0] *= 1.5
    theta[:, 1] *= 2.0
    theta[:, 2] *= 2.0
    kappa = 2.0
    p_off_prior = 0.2
    prior_off_classif = 0.2

    off_map_probs = np.ones(T) * p_off_prior

    # load/build reference map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, f'{ref_fname[:-4]}_wd_{w:.0f}.pickle')
    tstamps, poses, vo, descriptors = \
        load_subsampled_data(ref_traverse, ref_fname, pca_dim)
    xyzrpy = poses.to_xyzrpy()
    try:
        with open(fpath, "rb") as f:
            ref_map = pickle.load(f)
    except FileNotFoundError as err:
        print("cached map file not found, building map...")
        tstamps, poses, vo, _ = load_subsampled_data(ref_traverse, ref_fname, pca_dim)
        ref_map = build_map(ref_traverse, tstamps, poses, vo, descriptors,
                            w, 2 * args.subsample_threshold)
        with open(fpath, "rb") as f:
            pickle.dump(ref_map, f)
        print(f"successfully saved map as {fpath}")
    # add off map node
    ref_map.add_node("off")
    N = len(ref_map)

    # load reference descriptors
    descriptors = np.empty((N-1, len(ref_map.nodes[0]["nv"])))
    for i, data in ref_map.nodes.data():
        if i != 'off':
            descriptors[i] = data['nv']

    # load query sequence
    tstampsQ, posesQ, voQ, descriptorsQ = \
        load_subsampled_data(query_traverse, q_fname, pca_dim)
    xyzrpyQ = posesQ.to_xyzrpy()
    odomQ = voQ

    T = end_ind - start_ind
    odom = odomQ[start_ind:end_ind-1]
    sims = descriptorsQ[:, :pca_dim] @ descriptors[:, :pca_dim].T

    sims_viterbi = sims[start_ind:end_ind]
    # deviations_viterbi = [create_deviation_matrix(ref_map, o, Eoo, w) for
                  # o in tqdm(odom, desc='odom deviations')]

    # initialize prior belief (uniform)
    prior = np.ones(N) * (1. - p_off_prior) / (N - 1)
    prior[-1] = p_off_prior

    # # batch localization (entire sequence)

    # nv_lhoods_viterbi = vmflhood(sims_viterbi, kappa)
    # transition_matrices_viterbi = [create_transition_matrix(deviations_viterbi[t], N, Eoo,
                                                    # theta[t, 0], theta[t, 1],
                                                    # theta[t, 2])
                           # for t in range(T-1)]
    # alpha, off_lhoods, on_lhoods = forward_algorithm(
        # nv_lhoods_viterbi, transition_matrices_viterbi,
        # prior_off_classif, off_map_probs, prior
    # )
    # lhoods = np.concatenate((nv_lhoods_viterbi, off_lhoods[:, None]), axis=1)
    # state_seq = viterbi(lhoods, transition_matrices_viterbi, prior)
    # plot_viterbi(state_seq, poses, posesQ[start_ind:end_ind], sims_viterbi[start_ind:end_ind])

    # online filtering localizaton

    nv_lhoods = vmflhood(sims[start_ind:], kappa)
    t, ind = online_localization(odom[start_ind:], nv_lhoods,
                                 prior_off_classif, off_map_probs, prior,
                                 ref_map, Eoo, theta[0, 0], theta[0, 1], theta[0, 2],
                                 w)
    print(t, ind)

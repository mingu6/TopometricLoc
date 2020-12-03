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
import pickle
from scipy.sparse import csc_matrix
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra

from build_reference_map import read_descriptors, build_map, load_subsampled_data
from hmm_inference import viterbi, forward_algorithm, online_localization
from measurement_model import vmflhood
from motion_model import create_transition_matrix, create_deviation_matrix
from settings import DATA_DIR
import geometry

eval_lists = {}
eval_lists["night"] = np.array([[3000, 200],  # corner odom fucked, sticky, aliased
                                [0, 200],  # initial bend sticky, o/w good
                                [6000, 200],  # slight detour, bend, lil sticky
                                [4900, 200],  # around the two left turns bends
                                [4800, 200],  # aliased as shit...
                                [1500, 200],  # RHS sticky
                                [2100, 200],  # chuck a u
                                [400, 200]    # straight, aliased, sticky
                               ])

def save_deviations(dev, ref_traverse, q_traverse, ref_fname, q_fname):
    savedir = path.join(DATA_DIR, q_traverse, "localization",
                       f"{q_traverse}_{q_fname[:-4]}_{ref_traverse}_{ref_fname[:-4]}")
    os.makedirs(savedir, exist_ok=True)
    with open(path.join(savedir, "deviations.pickle"), 'wb') as f:
        pickle.dump(dev, f)
    return None


def load_deviations(ref_traverse, q_traverse, ref_fname, q_fname):
    devdir = path.join(DATA_DIR, q_traverse, "localization",
                       f"{q_traverse}_{q_fname[:-4]}_{ref_traverse}_{ref_fname[:-4]}")
    os.makedirs(devdir, exist_ok=True)
    with open(path.join(devdir, "deviations.pickle"), 'rb') as f:
        dev = pickle.load(f)
    return dev


def plot_viterbi(pred, start_ind, L, data, savedir):
    poses = data["poses"]
    posesQ = data["posesQ"][start_ind:start_ind+L]
    sims = data["sims"][start_ind:start_ind+L]
    # extract gt allocation of query poses
    gt = np.asarray([np.argmin(geometry.metric(Q, poses, w)) for Q in posesQ])
    # setup plots
    fig, ax = plt.subplots(1, 4, figsize=(18, 7))
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
        px = np.vstack((xyzrpyQ[t, 1], xyzrpy[pred[t], 1]))
        py = np.vstack((xyzrpyQ[t, 0], xyzrpy[pred[t], 0]))
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

    # zoomed in versions of above
    ax[2].set_title('Sequence localizer (Viterbi)')
    ax[2].scatter(xyzrpy[:, 1], xyzrpy[:, 0], color='black', s=5)
    ax[2].scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='green', s=5)
    ax[2].axis('square')
    for t in range(T):
        px = np.vstack((xyzrpyQ[t, 1], xyzrpy[pred[t], 1]))
        py = np.vstack((xyzrpyQ[t, 0], xyzrpy[pred[t], 0]))
        ax[2].plot(px, py, 'g-')
    ax[2].set_xlim(xyzrpyQ[:, 1].min(), xyzrpyQ[:, 1].max())
    ax[2].set_ylim(xyzrpyQ[:, 0].min(), xyzrpyQ[:, 0].max())

    ax[3].set_title('Single image retrieval')
    ax[3].scatter(xyzrpy[:, 1], xyzrpy[:, 0], color='black', s=5)
    ax[3].scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='red', s=5)
    ax[3].axis('square')
    for t in range(T):
        px = np.vstack((xyzrpyQ[t, 1], xyzrpy[IR_inds[t], 1]))
        py = np.vstack((xyzrpyQ[t, 0], xyzrpy[IR_inds[t], 0]))
        ax[3].plot(px, py, 'r-')
    ax[3].set_xlim(xyzrpyQ[:, 1].min(), xyzrpyQ[:, 1].max())
    ax[3].set_ylim(xyzrpyQ[:, 0].min(), xyzrpyQ[:, 0].max())
    # save plot
    fig.savefig(path.join(savedir, f'viterbi_matches_{start_ind}_{L}.png'))
    plt.close(fig)
    return None


def plot_online(pred, posterior, t, data, savedir):
    poses = data["poses"]
    posesQ = data["posesQ"][start_ind:start_ind+t]
    xyzrpy = poses.to_xyzrpy()
    xyzrpyQ = posesQ.to_xyzrpy()
    sims = data["sims"][start_ind:start_ind+t]
    # extract gt allocation of query poses
    gt = np.asarray([np.argmin(geometry.metric(Q, poses, w)) for Q in posesQ])
    # compute gt error
    terr, Rerr = (geometry.SE3.from_xyzrpy(xyzrpyQ[-1]) /
                  geometry.SE3.from_xyzrpy(xyzrpy[pred])).magnitude()
    # setup plots
    fig, ax = plt.subplots(1, 4, figsize=(18, 7))
    # plot batch state estimate from viterbi
    # true query pose (red) matched to best refn match
    T = len(posesQ)
    xyzrpy = poses.to_xyzrpy()
    xyzrpyQ = posesQ.to_xyzrpy()
    ax[0].set_title(f'Online localizer: err: {terr:.1f}m {Rerr*180/np.pi:.1f}deg, t = {t}')
    ax[0].scatter(xyzrpy[:, 1], xyzrpy[:, 0], c=posterior[:-1], s=5)
    ax[0].scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='green', s=5)
    ax[0].axis('square')
    # plot point of localization
    px = np.vstack((xyzrpyQ[-1, 1], xyzrpy[pred, 1]))
    py = np.vstack((xyzrpyQ[-1, 0], xyzrpy[pred, 0]))
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

    # zoomed in plots

    ax[2].set_title(f'Online localizer: err: {terr:.1f}m {Rerr*180/np.pi:.1f}deg, t = {t}')
    ax[2].scatter(xyzrpy[:, 1], xyzrpy[:, 0], c=posterior[:-1], s=5)
    ax[2].scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='green', s=5)
    ax[2].axis('square')
    # plot point of localization
    px = np.vstack((xyzrpyQ[-1, 1], xyzrpy[pred, 1]))
    py = np.vstack((xyzrpyQ[-1, 0], xyzrpy[pred, 0]))
    ax[2].plot(px, py, 'g-')
    ax[2].set_xlim(xyzrpyQ[:, 1].min()-20, xyzrpyQ[:, 1].max()+20)
    ax[2].set_ylim(xyzrpyQ[:, 0].min()-20, xyzrpyQ[:, 0].max()+20)

    ax[3].set_title('Single image retrieval')
    ax[3].scatter(xyzrpy[:, 1], xyzrpy[:, 0], color='black', s=5)
    ax[3].scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='red', s=5)
    ax[3].axis('square')
    for t in range(T):
        px = np.vstack((xyzrpyQ[t, 1], xyzrpy[IR_inds[t], 1]))
        py = np.vstack((xyzrpyQ[t, 0], xyzrpy[IR_inds[t], 0]))
        ax[3].plot(px, py, 'r-')
    ax[3].set_xlim(xyzrpyQ[:, 1].min()-20, xyzrpyQ[:, 1].max()+20)
    ax[3].set_ylim(xyzrpyQ[:, 0].min()-20, xyzrpyQ[:, 0].max()+20)
    # save plot
    fig.savefig(path.join(savedir, f'online_matches_{start_ind}.png'))
    plt.close(fig)
    return None


def vit(start_ind, length, data):
    """
    Convenience function, localize sequence given start index and length.
    data represents full query traverse info
    """
    # retrieve relevant data for localization
    nv_lhoods = data["nv_lhoods"][start_ind:start_ind+length, :]
    deviations = data["deviations"][start_ind:start_ind+length-1]
    Eoo = data["Eoo"]
    theta1 = data["theta1"]
    theta2 = data["theta2"]
    theta3 = data["theta3"]
    N = data["N"]
    prior = data["prior"]

    off_map_probs = np.ones(len(nv_lhoods)) * data["off_map_probs"]
    transition_matrices = [create_transition_matrix(deviations[t], N, Eoo,
                                                    theta1, theta2, theta3)
                           for t in range(length-1)]
    alpha, off_lhoods, on_lhoods = forward_algorithm(
        nv_lhoods, transition_matrices,
        prior_off_classif, off_map_probs, prior
    )
    lhoods = np.concatenate((nv_lhoods, off_lhoods[:, None]), axis=1)
    state_seq = viterbi(lhoods, transition_matrices, prior)
    return state_seq


def online(start_ind, data):
    """
    Convenience function, localize sequence given start index and length.
    data represents full query traverse info
    """
    # retrieve relevant data for localization
    nv_lhoods = data["nv_lhoods"][start_ind:]
    deviations = data["deviations"][start_ind:]
    Eoo = data["Eoo"]
    theta1 = data["theta1"]
    theta2 = data["theta2"]
    theta3 = data["theta3"]
    prior = data["prior"]
    xyzrpy = data["poses"].to_xyzrpy()

    off_map_probs = np.ones(len(deviations)) * data["off_map_probs"]
    t, ind, posterior = online_localization(deviations, nv_lhoods,
                                 prior_off_classif, off_map_probs, prior,
                                 Eoo, theta1, theta2, theta3, xyzrpy)
    return t, ind, posterior


def eval_sequence(start_ind, state_seq, data):
    T = len(state_seq)

    poses = data["poses"]
    posesQ = data["posesQ"][start_ind:start_ind+T]
    w = data["w"]

    gt_inds = np.asarray([np.argmin(geometry.metric(Q, poses, w)) for Q in posesQ])

    topo_err = np.abs(state_seq - gt_inds)
    precision = (topo_err <= 3).sum() / len(topo_err)
    avg_err = np.average(topo_err)
    max_err = topo_err.max()
    min_err = topo_err.min()

    return precision, avg_err, max_err, min_err


def eval_online(start_ind, t, proposal, data):
    poses = data["poses"]
    poseQ = data["posesQ"][start_ind+t]
    w = data["w"]

    gt_ind = np.argmin(geometry.metric(poseQ, poses, w))
    err = np.abs(proposal - gt_ind)

    return t, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-d", "--description", type=str, default="default",
                        help="description of model for experiment")
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
    parser.add_argument("-K", "--top-k", type=int, default=10, help="number of retrievals for obs lhood")
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
    theta[:, 0] *= 2.0
    theta[:, 1] *= 0.8
    theta[:, 2] *= 1.0
    kappa = 5.0
    p_off_prior = 0.2
    prior_off_classif = 0.2

    off_map_prob = 0.2

    # load/build reference map
    start = time.time()
    print("Loading reference map...")
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
    print(f"Done! duration {time.time() - start:.1f}s")
    # add off map node
    ref_map.add_node("off")
    N = len(ref_map)

    # load reference descriptors
    start = time.time()
    print("Loading reference descriptors...")
    descriptors = np.empty((N-1, len(ref_map.nodes[0]["nv"])))
    for i, data in ref_map.nodes.data():
        if i != 'off':
            descriptors[i] = data['nv']
    print(f"Done! duration {time.time() - start:.1f}s")

    # load query sequence
    start = time.time()
    print("Loading query data...")
    tstampsQ, posesQ, voQ, descriptorsQ = \
        load_subsampled_data(query_traverse, q_fname, pca_dim, ind=None)
    xyzrpyQ = posesQ.to_xyzrpy()
    odomQ = voQ
    print(f"Done! duration {time.time() - start:.1f}s")

    # compute image similarities and measurement likelihood
    start = time.time()
    print("Computing observation likelihoods...")
    sims = descriptorsQ[:, :pca_dim] @ descriptors[:, :pca_dim].T
    nv_lhoods = vmflhood(sims, kappa)
    print(f"Done! duration {time.time() - start:.1f}s")

    # deviations typically precomputed for whole query traverse, precompute
    # if file does not exist
    start = time.time()
    print("Loading deviations from disk...")
    try:
        deviations = load_deviations(ref_traverse, query_traverse, ref_fname, q_fname)
    except FileNotFoundError:
        print("Precomputed deviations not found, computing...")
        deviations = [create_deviation_matrix(ref_map, o, Eoo, w) for
                      o in tqdm(odomQ, desc='odom deviations')]
        save_deviations(deviations, ref_traverse, query_traverse, ref_fname, q_fname)
        print("Successfully saved deviations to disk!")
    print(f"Done! duration {time.time() - start:.1f}s")

    # initialize prior belief (uniform)
    prior = np.ones(N) * (1. - p_off_prior) / (N - 1)
    prior[-1] = p_off_prior

    # wrap all loaded data into dict for fast repeated inference
    data = {"nv_lhoods": nv_lhoods,
            "deviations": deviations,
            "Eoo": Eoo,
            "theta1": theta[0, 0],
            "theta2": theta[0, 1],
            "theta3": theta[0, 2],
            "off_map_probs": off_map_prob,
            "N": N,
            "prior": prior,
            "poses": poses,
            "posesQ": posesQ,
            "sims": sims,
            "w": w,
            "ref_map": ref_map}

    # batch localization (entire sequence)

    results_dir = path.join(DATA_DIR, query_traverse, "localization",
                            f"{query_traverse}_{q_fname[:-4]}_{ref_traverse}_{ref_fname[:-4]}")
    viterbi_dir = path.join(results_dir, "viterbi_figures", args.description)
    online_dir = path.join(results_dir, "online_figures", args.description)
    viterbi_results_dir = path.join(results_dir, "viterbi_results")
    online_results_dir = path.join(results_dir, "online_results")
    os.makedirs(viterbi_dir, exist_ok=True)
    os.makedirs(online_dir, exist_ok=True)
    os.makedirs(viterbi_results_dir, exist_ok=True)
    os.makedirs(online_results_dir, exist_ok=True)

    viterbi_results = []
    online_results = []

    for ev in tqdm(eval_lists[query_traverse]):
        start_ind = ev[0]
        T = ev[1]

        state_seq = vit(start_ind, T, data)
        plot_viterbi(state_seq, start_ind, T, data, viterbi_dir)

        # online filtering localizaton

        t, ind, posterior = online(start_ind, data)
        plot_online(ind, posterior, t, data, online_dir)

        # evaluate
        precision, avg_err, max_err, min_err = eval_sequence(start_ind, state_seq, data)
        t_loc, err = eval_online(start_ind, t, ind, data)
        viterbi_results.append([start_ind, T, precision, avg_err, max_err, min_err])
        online_results.append([start_ind, t_loc, err])

    # save localization results to file
    df_viterbi = pd.DataFrame(viterbi_results, columns=['start_ind', 'seq_length',
                                                        'seq_precision', 'avg_err',
                                                        'max_err', 'min_err'])
    df_online = pd.DataFrame(online_results, columns=['start_ind', 'steps_to_loc', 'err'])

    df_viterbi.to_csv(path.join(viterbi_results_dir, f"{args.description}.csv"), index=False)
    df_online.to_csv(path.join(online_results_dir, f"{args.description}.csv"), index=False)

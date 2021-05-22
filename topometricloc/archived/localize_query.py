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
from hmm_inference import forward_backward, Mstep, compute_objective, compute_objective_parts, highlevel_viterbi, forward_algorithm, viterbi
from measurement_model import vmflhood
from motion_model import create_transition_matrix, create_deviation_matrix
from settings import DATA_DIR
import geometry


def plot_viterbi(pred, gt, poses, posesQ):
    T = len(pred)
    xyzrpy = poses.to_xyzrpy()
    xyzrpyQ = posesQ.to_xyzrpy()
    plt.scatter(xyzrpy[:, 1], xyzrpy[:, 0], color='black', s=5)
    plt.scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], color='red', s=5)
    for t in range(T):
        px = np.vstack((xyzrpyQ[t, 1], xyzrpy[gt[t], 1]))
        py = np.vstack((xyzrpyQ[t, 0], xyzrpy[gt[t], 0]))
        plt.plot(px, py, 'r-')
    plt.show()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverse", type=str, default="night",
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str, required=True,
                    help="filename containing subsampled reference traverse poses")
    parser.add_argument("-qf", "--query-filename", type=str, required=True,
                    help="filename containing subsampled query traverse poses")
    parser.add_argument("-w", "--attitude-weight", type=float, default=10,
        help=("weight for attitude component of pose distances equal to d where"
              "1 / d being rotation angle (rad) equivalent to 1m translation"))
    parser.add_argument("-s", "--subsample-threshold", type=float, default=3,
        help=("distance travelled before adding a sequence element"))
    parser.add_argument("-p", "--pca-dim", type=int, default=1024,
                        help="number of dimensions for nv descriptor")
    args = parser.parse_args()

    w = args.attitude_weight;
    ref_traverse = args.reference_traverse
    query_traverse = args.query_traverse
    ref_fname = args.reference_filename
    q_fname = args.query_filename
    pca_dim = args.pca_dim

    att_wt = np.ones(6)
    att_wt[3:] *= w

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
    #odomQ = (posesQ[:-1] / posesQ[1:]).to_xyzrpy()
    odomQ = voQ

    # generate off-map probabilities
    off_map_probs = np.random.normal(0.2, 0.0, size=len(descriptorsQ))
    off_map_probs = np.clip(off_map_probs, 0., 1.)

    # off-map map modifications
    # gap_l = 705
    # gap_u = 715
    # for i in range(gap_l-10, gap_u):
        # edge_set = [edge for edge in ref_map.out_edges(i)]
        # for edge in edge_set:
            # if (edge[0] != edge[1] and edge[1] in np.arange(gap_l+1, gap_u+1)) \
                    # or (edge[0] != edge[1] and edge[0] in np.arange(gap_l, gap_u+1)):
                # ref_map.remove_edge(*edge)

    # plotting
    # fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    # fig.canvas.draw()

    # loop through sequence and update belief
    start_ind = 1675  # 550-580 hard 50-80 aliased
    start_ind = 100
    end_ind = len(descriptorsQ)
    end_ind = 130
    start_ind = 2400
    end_ind = 2440

    # f/w b/w algorithm variables
    T = end_ind - start_ind
    odom = odomQ[start_ind:end_ind-1]
    #print(odom[:, [0, 5]])
    #odom[10:20, :3] += np.random.normal(-0, 0.3, size=(10, 3))
    query_nv = descriptorsQ[start_ind:end_ind]
    #query_nv[20:30] += np.random.normal(0, 0.3, size=(10, 1024))

    pQ = posesQ[start_ind:end_ind]
    gt = np.asarray([np.argmin(geometry.metric(Q, poses, w)) for Q in pQ])

    # params
    Eoo = 0.7
    theta = np.ones((T - 1, 3))
    theta[:, 0] *= 1.5
    theta[:, 1] *= 2.0
    theta[:, 2] *= 2.0
    # theta[:, 0] *= 0.0
    # theta[:, 1] *= 0.0
    # theta[:, 2] *= 100.0
    lambda1 = 0.05
    kappa = np.ones(T) * 2.0
    p_off_prior = 0.2
    prior_off_classif = 0.2

    # process measurements
    sims = query_nv @ descriptors[:, :args.pca_dim].T
    deviations = [create_deviation_matrix(ref_map, o, Eoo, w) for
                  o in tqdm(odom, desc='odom deviations')]

    # import matplotlib.pyplot as plt
    # best_inds = np.argpartition(-sims, 20)[:, :20]
    # successes = np.asarray([np.any(geometry.metric(pq, poses[inds], 5.) < 10.) for pq, inds in zip(posesQ, best_inds)])
    # print("successes", successes.sum() / len(successes))

    # plt.scatter(xyzrpyQ[:, 1], xyzrpyQ[:, 0], c=successes[:])
    # plt.colorbar()
    # plt.show()

    # import matplotlib.pyplot as plt
    # plt.hist(deviations[0]["dev"][1], bins=100); plt.show()

    # initialize prior
    prior = np.ones(N) * (1. - p_off_prior) / (N - 1)
    prior[-1] = p_off_prior

    nv_lhoods = vmflhood(sims, kappa)
    transition_matrices = [create_transition_matrix(deviations[t], N, Eoo,
                                                    theta[t, 0], theta[t, 1],
                                                    theta[t, 2])
                           for t in range(T-1)]
    alpha, off_lhoods, on_lhoods = forward_algorithm(
        nv_lhoods, transition_matrices, prior_off_classif, off_map_probs, prior)
    lhoods = np.concatenate((nv_lhoods, off_lhoods[:, None]), axis=1)
    state_seq = viterbi(lhoods, transition_matrices, prior)
    print("pred:", state_seq)
    print("true:", gt)
    print("IR  :", np.argmax(sims, axis=1))
    plot_viterbi(state_seq, gt, poses, pQ)
    import pdb; pdb.set_trace()
    state_seq = highlevel_viterbi(
        on_lhoods, off_lhoods, transition_matrices, prior)

    niter = 20
    for i in range(niter):

        gamma, xi = forward_backward(sims, deviations, off_map_probs, prior,
                                     prior_off_classif, theta, kappa, Eoo, lambda1)
        ind_opt = np.argmin(geometry.metric(posesQ[start_ind+10], poses, 5))

        transition_matrices = create_transition_matrix(deviations[10], N, Eoo,
                                                        theta[10, 0], theta[10, 1],
                                                        theta[10, 2])

        import matplotlib.pyplot as plt
        print(gamma[10, -1], ind_opt)
        print(gamma[:, -1])
        plt.plot(np.arange(gamma.shape[1]-1), gamma[10, :-1])
        plt.plot(np.arange(ind_opt-10, ind_opt+10), gamma[10, ind_opt-10:ind_opt+10], color='red')
        plt.show()
        plt.scatter(xyzrpy[:, 1], xyzrpy[:, 0], c = gamma[10, :-1])
        #plt.scatter(xyzrpy[:, 1], xyzrpy[:, 0], c=deviations[10]["dev"][1])
        plt.colorbar()
        plt.scatter(xyzrpyQ[start_ind:end_ind, 1],
                    xyzrpyQ[start_ind:end_ind, 0], color='red')
        plt.show()
        plt.scatter(xyzrpy[:, 1], xyzrpy[:, 0], c = transition_matrices[:-1, -1].toarray()[:, 0])
        #plt.scatter(xyzrpy[:, 1], xyzrpy[:, 0], c=deviations[10]["dev"][1])
        plt.colorbar()
        plt.scatter(xyzrpyQ[start_ind:end_ind, 1],
                    xyzrpyQ[start_ind:end_ind, 0], color='red')
        plt.show()
        import pdb; pdb.set_trace()
        # obj = compute_objective(gamma, xi, prior, sims, deviations,
                                # theta, kappa, Eoo, lambda1)
        # obj1 = compute_objective_parts(gamma, xi, prior, sims, deviations,
                                # theta, kappa, Eoo, lambda1)

        print(f"start of iter {i+1}, marginal: {obj} {obj1}")
        #print(f"theta {theta} \nkappa {kappa} \n Eoo {Eoo}")
        #print(f"Eoo {Eoo}, theta1 {theta[:, 0]} theta2 {theta[:, 1]} theta3 {theta[:, 2]}")

        # maximization step

        theta, kappa, Eoo = Mstep(gamma, xi, prior, sims, deviations,
                                  theta, kappa, Eoo, lambda1)

        for t in range(T-1):
            from hmm_inference import l_theta3
            dev_within, opt_dev = deviations[t]["dev"]
            source_indices = deviations[t]["source"]
            dest_indices = deviations[t]["dest"]
            lt3 = np.zeros(100)
            for j, t3 in enumerate(np.linspace(0., 4., 100)):
                lt3[j] = l_theta3(np.array([t3]), xi[t], source_indices, dest_indices, dev_within)
            import matplotlib.pyplot as plt
            print(np.linspace(0, 4, 100)[np.argmin(lt3)])
            # plt.plot(np.linspace(0., 4., 100), lt3)
            # plt.pause(5)

        obj = compute_objective(gamma, xi, prior, sims, deviations,
                                theta, kappa, Eoo, lambda1)
        print(f"end of iter {i+1}, marginal: {obj}")
    import pdb; pdb.set_trace()

    for i in range(start_ind, end_ind):


        ax.scatter(xyzrpy[:, 1], xyzrpy[:, 0], c=belief[:-1])
        ax.set_title(f't={i-start_ind+1}, off map = {belief[-1]:.3f}')
        ax.scatter(xyzrpyQ[start_ind:i, 1], xyzrpyQ[start_ind:i, 0], color='red')
        plt.pause(0.1)

        import pdb; pdb.set_trace()

        ax.clear()

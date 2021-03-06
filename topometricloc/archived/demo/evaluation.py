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
import yaml

from build_reference_map import read_descriptors, build_map, load_subsampled_data
from hmm_inference import viterbi, constr_viterbi, forward_algorithm, online_localization
from measurement_model import vpr_lhood, off_map_prob, off_map_features
from motion_model import create_transition_matrix, odom_deviation
from settings import DATA_DIR
import geometry

self_dirpath = os.path.dirname(os.path.abspath(__file__))

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
eval_lists["dusk"] = np.array([[3000, 200],  # corner odom fucked, sticky, aliased
                                [0, 200],  # initial bend sticky, o/w good
                                [6000, 200],  # slight detour, bend, lil sticky
                                [4900, 200],  # around the two left turns bends
                                [4800, 200],  # aliased as shit...
                                [1500, 200],  # RHS sticky
                                [2100, 200],  # chuck a u
                                [400, 200]    # straight, aliased, sticky
                               ])
eval_lists["rain"] = np.array([[6440, 60], # minor detour corner
                               [3000, 200],  # corner odom fucked, sticky, aliased
                                [0, 200],  # initial bend sticky, o/w good
                                [6000, 200],  # slight detour, bend, lil sticky
                                [4900, 200],  # around the two left turns bends
                                [4800, 200],  # aliased as shit...
                                [1500, 200],  # RHS sticky
                                [2100, 200],  # chuck a u
                                [400, 200]    # straight, aliased, sticky
                               ])

eval_lists["rain_detour"] = np.array([[4850, 200], # minor detour corner
                               [6050, 200],  # corner odom fucked, sticky, aliased
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
                       f"{q_traverse}_{q_fname[:-4]}_{ref_traverse}_{ref_fname[:-7]}")
    os.makedirs(savedir, exist_ok=True)
    with open(path.join(savedir, "deviations.pickle"), 'wb') as f:
        pickle.dump(dev, f)
    return None


def load_deviations(ref_traverse, q_traverse, ref_fname, q_fname):
    devdir = path.join(DATA_DIR, q_traverse, "localization",
                       f"{q_traverse}_{q_fname[:-4]}_{ref_traverse}_{ref_fname[:-7]}")
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
    ax[0].scatter(xyzrpyQ[pred == -1, 1],
                  xyzrpyQ[pred == -1, 0], color='red', s=5)
    ax[0].axis('square')
    for t in range(T):
        if pred[t] != -1:
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
    ax[2].scatter(xyzrpyQ[pred == -1, 1],
                  xyzrpyQ[pred == -1, 0], color='red', s=5)
    ax[2].axis('square')
    for t in range(T):
        if pred[t] != -1:
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
    if start_ind in [4850, 6050, 1500]:
        plt.show()
        import pdb; pdb.set_trace()
    else:
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
    theta = data["theta"]
    p_min = data["p_min"]
    p_max = data["p_max"]
    d_min = data["d_min"]
    d_max = data["d_max"]
    prior = data["prior"]
    width = data["width"]
    N = data["N"]
    prior = data["prior"]
    off_map_probs = data["off_map_probs"][start_ind:start_ind+length]

    # synthetic classifier
    if start_ind == 6050:
        i1 = 75
        i2 = 180
    elif start_ind == 4850:
        i1 = 60
        i2 = 140
    off_map_probs = np.random.binomial(1, 0.2, size=len(off_map_probs)).astype(np.float)
    if start_ind in [6050, 4850]:
        off_map_probs[i1:i2] = np.random.binomial(1, 0.8, size=i2-i1)
    off_map_probs[off_map_probs == 0] = 0.4
    off_map_probs[off_map_probs == 1] = 0.6

    params = {"Eoo": Eoo, "theta": theta, "N": N, "p_min": p_min, "p_max": p_max,
              "d_min": d_min, "d_max": d_max, "width": width}

    transition_matrices = [create_transition_matrix(deviations[t], params)
                           for t in range(length-1)]
    alpha, off_lhoods, on_lhoods, agg_Es = forward_algorithm(
        nv_lhoods, transition_matrices,
        prior_off_classif, off_map_probs, prior
    )
    lhoods = np.concatenate((nv_lhoods, off_lhoods[:, None]), axis=1)
    agg_lhoods = np.vstack((on_lhoods, off_lhoods)).T
    agg_prior = np.array([1. - Eoo, Eoo])
    state_seq_hl = viterbi(agg_lhoods, agg_Es, agg_prior)
    state_seq = constr_viterbi(lhoods, transition_matrices, prior, state_seq_hl)
    #import pdb; pdb.set_trace()
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
    theta = data["theta"]
    p_min = data["p_min"]
    p_max = data["p_max"]
    d_min = data["d_min"]
    d_max = data["d_max"]
    prior = data["prior"]
    width = data["width"]
    xyzrpy = data["poses"].to_xyzrpy()

    off_map_probs = data["off_map_probs"]
    t, ind, posterior = online_localization(deviations, nv_lhoods,
                                 prior_off_classif, off_map_probs, prior,
                                 Eoo, theta, p_min, p_max, d_min, d_max, width,
                                            xyzrpy)
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
    parser.add_argument("-rf", "--reference-filename", type=str, default='t_1_w_10_wd_8.pickle',
                    help="filename containing subsampled reference traverse poses")
    parser.add_argument("-qf", "--query-filename", type=str, default='t_1_w_10.csv',
                    help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="default.yaml",
                        help="filename containing model parameters")
    args = parser.parse_args()

    # configuration variables

    ref_traverse = args.reference_traverse
    query_traverse = args.query_traverse
    ref_fname = args.reference_filename
    q_fname = args.query_filename

    # parameters
    params_path = path.abspath(path.join(self_dirpath, "..", "params"))
    with open(path.join(params_path, args.params), 'r') as f:
        params = yaml.load(f)

    w = params["motion"]["att_wt"]
    Eoo = params["motion"]["p_off_off"]
    theta = params["motion"]["theta"]
    p_min = params["motion"]["p_off_min"]
    p_max = params["motion"]["p_off_max"]
    d_min = params["motion"]["d_min"]
    d_max = params["motion"]["d_max"]
    p_off_prior = params["init"]["prior_off"]
    prior_off_classif = params["measurement"]["p_off_prior"]
    lhmax = params["measurement"]["max_lvl"]
    lvl = params["measurement"]["min_lvl"]
    alpha = params["measurement"]["alpha"]
    pca_dim = params["measurement"]["pca_dim"]
    k = params["measurement"]["k"]
    p_min_meas = params["measurement"]["p_off_min"]
    p_max_meas = params["measurement"]["p_off_max"]

    att_wt = np.ones(6)
    att_wt[3:] *= w

    # load/build reference map
    start = time.time()
    print("Loading reference map...")
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, ref_fname)
    spl = ref_fname.split("_")  # retrieve raw poses
    subsampled_fname = "_".join(spl[:4]) + ".csv"
    tstamps, poses, vo, descriptors = \
        load_subsampled_data(ref_traverse, subsampled_fname, pca_dim)
    xyzrpy = poses.to_xyzrpy()
    try:
        with open(fpath, "rb") as f:
            ref_map = pickle.load(f)
    except FileNotFoundError as err:  # NOTE: THIS DOES NOT WORK, JUST BUILD THE MAP FIRST
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
    wd = int(spl[-1][:-7])  # number of transitions per node (width)

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
    nv_lhoods = vpr_lhood(sims, lhmax, lvl, alpha, k)
    off_features = off_map_features(sims, k)
    off_map_probs = off_map_prob(off_features, p_min_meas, p_max_meas)
    np.clip(off_map_probs, 0.4, 0.6, out=off_map_probs)
    # off_map_probs[off_map_probs >= 0.5] = 0.6
    # off_map_probs[off_map_probs < 0.5] = 0.4
    print(f"Done! duration {time.time() - start:.1f}s")

    # deviations typically precomputed for whole query traverse, precompute
    # if file does not exist
    start = time.time()
    print("Loading deviations from disk...")
    try:
        deviations = load_deviations(ref_traverse, query_traverse, ref_fname, q_fname)
    except FileNotFoundError:
        print("Precomputed deviations not found, computing...")
        deviations = [odom_deviation(ref_map, o, w) for
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
            "theta": theta,
            "p_min": p_min,
            "p_max": p_max,
            "d_min": d_min,
            "d_max": d_max,
            "off_map_probs": off_map_probs,
            "N": N,
            "prior": prior,
            "poses": poses,
            "posesQ": posesQ,
            "sims": sims,
            "w": w,
            "width": wd,
            "ref_map": ref_map,
            "off_map_feats": off_features}

    # batch localization (entire sequence)

    results_dir = path.join(DATA_DIR, query_traverse, "localization",
                            f"{query_traverse}_{q_fname[:-4]}_{ref_traverse}_{ref_fname[:-7]}")
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
        #import pdb; pdb.set_trace()
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

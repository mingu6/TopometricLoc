import argparse
import importlib
import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

import yaml

from data.utils import load_pose_data, read_global, read_local_raw
from ours.mapping import RefMap
from results_batch import precision_at_tol
from evaluation_batch import t_off_thres, R_off_thres, pose_and_off_map_err, id_off_map_queries
from settings import DATA_DIR
from utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))


def vis_batch(ax, query_gt, refMap, pred_inds, query_off_flags, t_thres, R_thres,
              method):
    T = len(query_gt)
    pred_off_mask = pred_inds == -1
    pred_on_mask = pred_inds != -1
    t_err, R_err = pose_and_off_map_err(gt, refMap, pred_inds,
                                        query_off_flags)
    success = np.logical_and(t_err < t_thres, R_err < R_thres)
    failure = np.logical_not(success)
    # plot ground truth query and map
    ax.scatter(refMap.gt_poses[:, 1], refMap.gt_poses[:, 0],
               color='black', label='Map GT')
    # plot correctly localized points in sequence within tolerance
    on_success = np.logical_and(success, pred_on_mask)
    ax.scatter(query_gt[on_success, 1], query_gt[on_success, 0],
               color='green', label="On Success")
    on_success_ind = np.squeeze(np.argwhere(on_success))
    for t in on_success_ind:
        px = np.vstack((query_gt[t, 1], refMap.gt_poses[pred_inds[t], 1]))
        py = np.vstack((query_gt[t, 0], refMap.gt_poses[pred_inds[t], 0]))
        ax.plot(px, py, 'g-')
    # plot localized points in sequence outside tolerance
    on_failure = np.logical_and(failure, pred_on_mask)
    ax.scatter(query_gt[on_failure, 1], query_gt[on_failure, 0],
               color='brown', label="On Failure")
    on_failure_ind = np.squeeze(np.argwhere(on_failure))
    for t in on_failure_ind:
        px = np.vstack((query_gt[t, 1], refMap.gt_poses[pred_inds[t], 1]))
        py = np.vstack((query_gt[t, 0], refMap.gt_poses[pred_inds[t], 0]))
        ax.plot(px, py, color='brown')
    # predicted off-map when on map
    pred_off_but_on = np.logical_and(pred_off_mask, np.logical_not(query_off_flags))
    ax.scatter(query_gt[pred_off_but_on, 1], query_gt[pred_off_but_on, 0],
               color='pink', label="On Failure (off)")
    # plot correct off-map association
    off_success = np.logical_and(query_off_flags, pred_off_mask)
    ax.scatter(query_gt[off_success, 1], query_gt[off_success, 0],
               color='yellow', label="Off Success")
    # plot incorrect off-map association
    off_failure = np.logical_and(query_off_flags, pred_on_mask)
    ax.scatter(query_gt[off_failure, 1], query_gt[off_failure, 0],
               color='red', label="Off Failure")
    ax.set_title(method)
    ax.legend()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run localization experiments for our method or comparisons"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverse", type=str,
                        default="sun_clouds_detour2",
                        choices=["sun_clouds_detour2", "night", "overcast_detour2"],
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str,
                        default='t_1_w_10_wd_2.pickle',
                        help="filename containing reference map object")
    parser.add_argument("-qf", "--query-filename", type=str, default='t_1_w_10.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-t", "--trials", type=int, default=100,
                        help="number of trials to run, evenly distributed")
    parser.add_argument("-ll", "--length-low", type=int, default=150,
                        help="minimum sequence length")
    parser.add_argument("-lh", "--length-high", type=int, default=250,
                        help="maximum sequence length")
    parser.add_argument("-m", "--methods", type=str, nargs="+",
                        choices=["ours", "xu20", "stenborg20"],
                        default=["ours", "xu20", "stenborg20"])
    parser.add_argument("-te", "--transl-err", type=float, default=5.,
                        help="translational error (m) threshold for success")
    parser.add_argument("-Re", "--rot-err", type=float, default=30.,
                        help="rotational error threshold (deg) for success")
    args = parser.parse_args()

    ref_traverse = args.reference_traverse
    query = args.query_traverse
    r_fname = args.reference_filename
    q_fname = args.query_filename

    # load map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, r_fname)
    with open(fpath, "rb") as f:
        refMap = pickle.load(f)

    # set sequence lengths
    np.random.seed(10)
    seq_len = np.random.randint(args.length_low, args.length_high,
                                args.trials)

    # load query sequence
    tstampsQ, xyzrpyQ, odomQ = load_pose_data(query, q_fname)
    query_global = read_global(query, tstampsQ)
    # label query poses as off-map or on-map based on nearest ref
    query_off_flags = id_off_map_queries(xyzrpyQ, refMap.gt_poses,
                                         t_off_thres, R_off_thres)
    # subsample query traverse for starting point, leave gap at the end
    # of traverse so last trials has some steps to converge!
    start_inds = np.linspace(0, len(tstampsQ) - args.length_high - 1,
                             args.trials).astype(int)

    # load map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, r_fname)
    with open(fpath, "rb") as f:
        refMap = pickle.load(f)

    # load query sequence
    tstampsQ, xyzrpyQ, odomQ = load_pose_data(query, q_fname)
    query_global = read_global(query, tstampsQ)

    # subsample query traverse for starting point
    start_inds = np.linspace(0, len(tstampsQ) - args.length_high - 1,
                             args.trials).astype(int)

    # enter trial number to visualize
    while True:
        user = input(f"Enter trial number /{len(start_inds)} (type q to quit): ")
        try:
            trial_no = int(user)
            if trial_no not in range(1, len(start_inds) + 1):
                print(f"Invalid value {user}! Enter an integer between "
                      f"1 and {len(start_inds)} or 'q' to quit.")
                continue
            else:
                sInd = start_inds[trial_no-1]
        except ValueError as e:
            if user == 'q':
                sys.exit("Exiting...")
            else:
                print(e)
                print(f"Invalid value {user}! Enter an integer between 1 "
                      f"and {len(start_inds)} or 'q' to quit.")
                continue

        # setup plots
        fig, axs = plt.subplots(1, len(args.methods))

        for i, method in enumerate(args.methods):

            # for each method (ours/comparisons), import assoc. module

            if method == "ours":
                from ours.batch import batchlocalization
            else:
                batch = importlib.import_module(
                    f"comparison_methods.{method}.batch")
                batchlocalization = batch.batchlocalization

            # import params

            if args.params:
                params_file = args.params
            else:
                params_file = method + ".yaml"

            # read in parameters

            params_path = path.abspath(path.join(self_dirpath, "params"))
            with open(path.join(params_path, params_file), 'r') as f:
                params = yaml.safe_load(f)

            # start localization process

            results = []
            ntrials = len(start_inds)

            T = seq_len[trial_no-1]
            qOdoms = odomQ[sInd:sInd+T].copy()
            qGlbs = query_global[sInd:sInd+T+1].copy()
            qLocs = [read_local_raw(query, tstampsQ[s]) for s in
                     range(sInd, sInd+T+1)]
            gt = xyzrpyQ[sInd:sInd+T+1].copy()
            # run batch localization
            pred_inds = batchlocalization(params, refMap, qOdoms,
                                          qGlbs, qLocs)
            # evaluate error between estimated nodes and gt poses
            t_err, R_err = pose_and_off_map_err(gt, refMap, pred_inds,
                                                query_off_flags[sInd:sInd+T+1])
            success = np.logical_and(t_err < args.transl_err,
                                     R_err < args.rot_err)
            precision = success.sum() / len(success) * 100.
            print(f"Trial: {trial_no}, Length: {T}, Method: {method}, "
                  f"Precision: {precision:.1f}")

            vis_batch(axs[i], gt, refMap, pred_inds,
                      query_off_flags[sInd:sInd+T+1],
                      args.transl_err, args.rot_err, method)
        plt.show()

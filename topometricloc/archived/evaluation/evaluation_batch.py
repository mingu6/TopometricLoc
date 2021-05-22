import argparse
import importlib
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm

import csv
import yaml

from topometricloc.data.utils import load_pose_data, read_global, read_local_raw
from topometricloc.ours.mapping import RefMap
from topometricloc.settings import DATA_DIR, RESULTS_DIR
from topometricloc.utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))

##### HARDCODED THRESHOLDS FOR WHAT DEFINES AN OFF-MAP QUERY #####
t_off_thres = 5.
R_off_thres = 30.
##################################################################


def id_off_map_queries(query_gt, ref_gt, t_off_thres, R_off_thres):
    """
    Identifies which query poses are considered as "off-map" or "within-map"
    given tolerances on translation and orientation error.
    Args:
        query_gt: Nqx3 set of query ground truth poses
        ref_gt: Nx3 set of reference map ground truth poses
        t_off_thres: threshold on minimum translation error for query to be off
        R_off_thres: threshold on minimum orientation error for query to be off
    returns:
        off_on_id: length Nq array with False (on) and True (off)
    """
    # compute error between all query/ref pose pairs
    t_err_mat, R_err_mat = pose_err(query_gt, ref_gt, degrees=True)
    within = (t_err_mat < t_off_thres) * (R_err_mat < R_off_thres)
    off_on_id = np.logical_not(np.any(within, axis=1))
    return off_on_id


def pose_and_off_map_err(query_gt, refMap, pred_inds, query_off_flags):
    """
    Compute error between query and predicted reference sequence, accounting
    for off-map nodes. Nodes correctly classified as off-map get 0 error
    for translation and orientation and 1000. error for incorrect off-map
    allocation.
    Args:
        query_gt: Tx3 query grount truth poses for sequence
        refMap: reference map object
        pred_inds: predicted reference map indices for query sequence
        query_off_flags: length T boolean array with query true
                         on/off status (off = True, on = False)
    Returns:
        t_err: translation error for length T sequence
        R_err: orientation error for length T sequence
    """
    assert len(query_gt) == len(pred_inds)
    # off-map state is last state, rebrand to -1 for masking later
    off_pred_inds = pred_inds == refMap.N
    pred_inds[off_pred_inds] = -1
    ref_gt = refMap.gt_poses[pred_inds]
    # recover error between pred and query gt, pose_err returns pairwise error
    t_err_mat, R_err_mat = pose_err(query_gt, ref_gt, degrees=True)
    t_err = np.diagonal(t_err_mat).copy()
    R_err = np.diagonal(R_err_mat).copy()

    # off-map errors will be incorrect, rewrite depending on correct off-map

    # if model correctly associates off, set 0 error
    off_correct = off_pred_inds * query_off_flags
    t_err[off_correct] = 0.
    R_err[off_correct] = 0.
    # if model incorrectly allocates off when not, set 100. error
    off_incorrect = off_pred_inds * np.logical_not(query_off_flags)
    t_err[off_incorrect] = 100.
    R_err[off_incorrect] = 100.
    return t_err, R_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run localization experiments for our method or comparisons"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverses", type=str, nargs="+",
                        default=["sun_clouds_detour2", "night", "rain_detour"],
                        choices=["sun_clouds_detour2", "night", "rain_detour"],
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str,
                        default='xy_1_t_10_wd_4.pickle',
                        help="filename containing reference map object")
    parser.add_argument("-qf", "--query-filename", type=str, default='xy_3_t_15.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-t", "--trials", type=int, default=100,
                        help="number of trials to run, evenly distributed")
    parser.add_argument("-ll", "--length-low", type=int, default=150,
                        help="minimum sequence length")
    parser.add_argument("-lh", "--length-high", type=int, default=250,
                        help="maximum sequence length")
    parser.add_argument("-m", "--methods", nargs="+", type=str,
                        choices=["ours", "xu20", "stenborg20"],
                        default=["ours", "xu20", "stenborg20"])
    args = parser.parse_args()

    ref_traverse = args.reference_traverse
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

    pbarq = tqdm(args.query_traverses)
    for query in pbarq:
        pbarq.set_description(query)
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

        pbarm = tqdm(args.methods)
        for method in pbarm:
            pbarm.set_description(method)

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

            # create description of experiment if not specified

            description = \
                f"{ref_traverse}_{r_fname[:-7]}_{query}_{q_fname[:-4]}_{method}"

            # start localization process

            results = []
            ntrials = len(start_inds)

            for i, sInd in enumerate(tqdm(start_inds, desc="trials",
                                          leave=False)):
                trial_no = i + 1
                T = seq_len[i]
                qOdoms = odomQ[sInd:sInd+T].copy()
                qGlbs = query_global[sInd:sInd+T+1].copy()
                qLocs = [read_local_raw(query, tstampsQ[s]) for s in
                         range(sInd, sInd+T+1)]
                gt = xyzrpyQ[sInd:sInd+T+1].copy()
                # run batch localization
                pred_inds = batchlocalization(params, refMap, qOdoms,
                                              qGlbs, qLocs)
                # evaluate error between estimated nodes and gt poses
                t_err, R_err = pose_and_off_map_err(gt, refMap,
                                                    pred_inds,
                                                    query_off_flags[sInd:sInd+T+1])
                trial_results = []
                for t in range(T+1):
                    result = (t_err[t], R_err[t])
                    trial_results.extend(result)
                results.append([trial_no, T, *trial_results])

            # create new folder to store results

            rpath = path.join(RESULTS_DIR, "batch")
            os.makedirs(rpath, exist_ok=True)  # create base results folder if required
            trials = [int(p.split("_")[-1]) for p in os.listdir(rpath)
                      if "_".join(p.split("_")[:-1]) == description]
            if len(trials) > 0:
                trial_count = max(trials) + 1
                results_path = path.join(rpath, f"{description}_{trial_count}")
            else:
                results_path = path.join(rpath, f"{description}_1")
            os.makedirs(results_path)

            # write results to csv

            with open(path.join(results_path, "results.csv"), "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                # write header row
                t_max = np.max(seq_len)
                header = ["trial", "length",
                          *(f"t_err_{t},R_err_{t}" for t in range(t_max+1))]
                csv_file.write(",".join(header) + "\n")
                for result in results:
                    writer.writerow(result)

            # save parameters into results folder for records

            with open(path.join(results_path, 'params.yaml'), 'w') as f:
                yaml.dump(params, f)

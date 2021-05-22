import argparse
import importlib
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm

import yaml
import scipy.sparse as sparse
from scipy.special import logsumexp

from data.utils import load_pose_data, read_global, read_local_raw
from geometry import SE2
from ours.mapping import RefMap
from results.run_loop_closure import on_xy_thres, on_rot_thres
from settings import DATA_DIR, RESULTS_DIR
from utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run loop closure experiments for our method or comparisons"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverses", type=str, nargs="+", required=True,
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str,
                        default='xy_1_t_10_wd_4.pickle',
                        help="filename containing reference map object")
    parser.add_argument("-qf", "--query-filename", type=str, default='xy_3_t_15.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    args = parser.parse_args()

    ref_traverse = args.reference_traverse
    r_fname = args.reference_filename
    q_fname = args.query_filename

    # load map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, r_fname)
    with open(fpath, "rb") as f:
        refMap = pickle.load(f)

    pbarq = tqdm(args.query_traverses)
    for query in pbarq:
        pbarq.set_description(query)
        # load query sequence
        tstampsQ, gtQ, muQ, SigmaQ = load_pose_data(query, q_fname)
        query_global = read_global(query, tstampsQ)


        # for each method (ours/comparisons), import assoc. module

        from comparison_methods.baseline.localization import Localization

        # import params

        if args.params:
            params_file = args.params
        else:
            params_file = "baseline.yaml"

        # read in parameters

        params_path = path.abspath(path.join(self_dirpath, "..", "params"))
        with open(path.join(params_path, params_file), 'r') as f:
            params = yaml.safe_load(f)

        # create description of experiment if not specified

        description = \
            f"{ref_traverse}_{r_fname[:-7]}_{query}_{q_fname[:-4]}_baseline"

        # identify on-map status of query

        gtQSE2 = SE2(gtQ)
        refgtSE2 = SE2(refMap.gt_poses)

        def min_pose_err(poseSE2):
            xy_errs, rot_errs = (poseSE2 / refgtSE2).magnitude()
            wgtd = xy_errs + 10. * rot_errs
            best = np.argmin(wgtd)
            return xy_errs[best], rot_errs[best]
        qgt_err = np.asarray(list(map(min_pose_err, gtQSE2))).T
        qxy, qrot = qgt_err

        # night-rain traverses uses GPS, orientation wrong so override

        on_xy_thres1 = on_xy_thres if query != 'night-rain' else 10.
        on_rot_thres1 = on_rot_thres if query != 'night-rain' else 360.
        q_on_map = np.logical_and(qxy < on_xy_thres1,
                                  qrot * 180. / np.pi < on_rot_thres1)

        # save results
        checks = []
        xy_errs = []
        rot_errs = []
        ref_inds = []
        on_map_stats = []

        # setup localization object
        loc = Localization(params, refMap)
        for i in tqdm(range(len(tstampsQ)), desc="extract", leave=False):

            qLoc = read_local_raw(query, tstampsQ[i])
            qGlb = query_global[i]
            qmu, qSigma = muQ[i], SigmaQ[i]
            # usually at t=0 there is a meas. update with no motion
            # separate initialization performed
            if i == 0:
                loc.init(qmu, qSigma, qGlb, qLoc)
            else:
                # update state estimate
                loc.update(qmu, qSigma, qGlb, qLoc)
            pred_ind, check, score = loc.converged(qGlb, qLoc)
            xy_err, rot_err = pose_err(gtQ[i], refMap.gt_poses[pred_ind],
                                       degrees=True)
            checks.append(check)
            ref_inds.append(pred_ind)
            xy_errs.append(xy_err)
            rot_errs.append(rot_err)

        checks = np.asarray(checks)
        ref_inds = np.asarray(ref_inds)
        xy_errs = np.asarray(xy_errs)
        rot_errs = np.asarray(rot_errs)

        results = {"checks": checks,
                   "ref_inds": ref_inds,
                   "xy_err": xy_errs,
                   "rot_err": rot_errs,
                   "on_status": q_on_map}

        # create new folder to store results

        rpath = path.join(RESULTS_DIR, "loop_closure")
        os.makedirs(rpath, exist_ok=True)  # create base results folder if required
        trials = [int(p.split("_")[-1]) for p in os.listdir(rpath)
                  if "_".join(p.split("_")[:-1]) == description]
        if len(trials) > 0:
            trial_count = max(trials) + 1
            results_path = path.join(rpath, f"{description}_{trial_count}")
        else:
            results_path = path.join(rpath, f"{description}_2")
        os.makedirs(results_path)

        # dump results

        with open(path.join(results_path, "results.pickle"), "wb") as f:
            pickle.dump(results, f)

        # save parameters into results folder for records

        with open(path.join(results_path, 'params.yaml'), 'w') as f:
            yaml.dump(params, f)

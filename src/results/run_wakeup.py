import argparse
import importlib
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm

import yaml

from data.utils import load_pose_data, read_global, read_local_raw
from geometry import SE2
from ours.mapping import RefMap
from settings import DATA_DIR, RESULTS_DIR
from utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))

# off-map thresholds, within this tolerance to be considered on-map
on_xy_thres = 5.
on_rot_thres = 30.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run localization experiments for our method or comparisons"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverses", type=str, nargs="+", required=True,
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str,
                        default='xy_1_t_10_wd_4.pickle',
                        help="filename containing reference map object")
    parser.add_argument("-qf", "--query-filename", type=str, default='xy_2_t_15.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-n", "--num-trials", type=int, default=500,
                        help="number of trials to run, evenly distributed")
    parser.add_argument("-m", "--methods", nargs="+", type=str,
                        choices=["ours", "xu20", "stenborg20", "baseline", "noverif", "nooff"],
                        default=["ours", "xu20", "stenborg20", "baseline", "noverif", "nooff"])
    parser.add_argument("-d", "--max-dist", type=float, default=200,
                        help="maximum distance travelled (m) per trial")
    args = parser.parse_args()

    ref_traverse = args.reference_traverse
    r_fname = args.reference_filename
    q_fname = args.query_filename
    max_dist = args.max_dist

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

        # compute distance travelled (meters) from starting location
        # for all query images in sequence

        gtQSE2 = SE2(gtQ)
        relpose = (gtQSE2[:-1] / gtQSE2[1:]).to_vec()
        relpose = np.vstack((np.zeros(3), relpose))
        dist_from_start = np.cumsum(np.linalg.norm(relpose[:, :2], axis=1))

        # subsample query traverse for starting point, allow for gap of
        # max dist at end of traverse for final trial

        ind_max = np.max(np.argwhere(dist_from_start[-1] -
                                     dist_from_start > max_dist))
        start_inds = np.linspace(0, ind_max, args.num_trials).astype(int)

        # identify on-map status of query

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

        pbarm = tqdm(args.methods)
        for method in pbarm:
            pbarm.set_description(method)

            # for each method (ours/comparisons), import assoc. module

            if method == "ours":
                from ours.localization import LocalizationFull as Localization
            elif method == "noverif":
                from ours.localization import LocalizationNoVerif as Localization
            elif method == "nooff":
                from ours.localization import LocalizationNoOff as Localization
            else:
                localization= importlib.import_module(
                    f"comparison_methods.{method}.localization")
                Localization = localization.Localization

            # import params

            if args.params:
                params_file = args.params
            else:
                param_fname = "ours" if method in ['noverif', 'nooff'] else method
                params_file = param_fname + ".yaml"

            # read in parameters

            params_path = path.abspath(path.join(self_dirpath, "..", "params"))
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
                # setup localization object
                loc = Localization(params, refMap)

                # save results
                scores = []
                checks = []
                xy_errs = []
                rot_errs = []
                ref_inds = []
                on_map_stats = []
                off_probs = []

                s = sInd
                while dist_from_start[s] - dist_from_start[sInd] < max_dist:
                    t = s - sInd  # trial timestep
                    qLoc = read_local_raw(query, tstampsQ[s])
                    qGlb = query_global[s]
                    qmu, qSigma = muQ[s], SigmaQ[s]
                    # usually at t=0 there is a meas. update with no motion
                    # separate initialization performed
                    if t == 0:
                        loc.init(qmu, qSigma, qGlb, qLoc)
                    else:
                        # update state estimate
                        loc.update(qmu, qSigma, qGlb, qLoc)
                    # check convergence
                    ind_pred, check, score = loc.converged(qGlb, qLoc)
                    scores.append(score)
                    checks.append(check)
                    ref_inds.append(ind_pred)
                    off_prob = loc.belief[-1] if method in ["ours", "noverif"] else 0.
                    off_probs.append(off_prob)
                    # evaluation against ground truth
                    xy_err, rot_err = pose_err(gtQ[sInd+t], refMap.gt_poses[ind_pred],
                                               degrees=True)
                    xy_errs.append(xy_err)
                    rot_errs.append(rot_err)
                    on_map_stats.append(q_on_map[s])
                    s += 1

                result = {"dist": dist_from_start[sInd:s] - dist_from_start[sInd],
                          "scores": np.asarray(scores), "checks": np.asarray(checks),
                          "ref_inds": np.asarray(ref_inds), "off_probs": np.asarray(off_probs),
                          "xy_err": np.asarray(xy_errs),
                          "rot_err": np.asarray(rot_errs),
                          "on_status": np.asarray(on_map_stats)}

                trial_no = i + 1
                results.append(result)

            # create new folder to store results

            rpath = path.join(RESULTS_DIR, "wakeup")
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

import argparse
import importlib
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm

import csv
import yaml

from data.utils import load_pose_data, read_global, read_local_raw
from ours.mapping import RefMap
from settings import DATA_DIR, RESULTS_DIR
from utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run localization experiments for our method or comparisons"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverses", type=str, nargs="+",
                        default=["sun_clouds_detour2", "night"],
                        choices=["sun_clouds_detour2", "night"],
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str,
                        default='t_1_w_10_wd_2.pickle',
                        help="filename containing reference map object")
    parser.add_argument("-qf", "--query-filename", type=str, default='t_1_w_10.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-t", "--trials", type=int, default=1000,
                        help="number of trials to run, evenly distributed")
    parser.add_argument("-m", "--methods", nargs="+", type=str,
                        choices=["ours", "xu20", "stenborg20", "baseline"],
                        default=["ours", "xu20", "stenborg20", "baseline"])
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
        tstampsQ, xyzrpyQ, odomQ = load_pose_data(query, q_fname)
        query_global = read_global(query, tstampsQ)
        # compute distance travelled (meters) from starting location
        # for all query images in sequence
        relpose = xyzrpyQ[1:, :2] - xyzrpyQ[:-1, :2]
        relpose = np.vstack((np.zeros(2), relpose))
        dist_from_start = np.cumsum(np.linalg.norm(relpose, axis=1))

        # subsample query traverse for starting point, leave gap at the end
        # of traverse so last trials has some steps to converge!
        start_inds = np.linspace(0, len(tstampsQ) - 50, args.trials).astype(int)

        pbarm = tqdm(args.methods)
        for method in pbarm:
            pbarm.set_description(method)

            # for each method (ours/comparisons), import assoc. module

            if method == "ours":
                from ours.online import OnlineLocalization
            else:
                online = importlib.import_module(
                    f"comparison_methods.{method}.online")
                OnlineLocalization = online.OnlineLocalization

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
            t_max = 0
            ntrials = len(start_inds)

            for i, sInd in enumerate(tqdm(start_inds, desc="trials",
                                          leave=False)):
                trial_results = []
                # setup localization object
                loc = OnlineLocalization(params, refMap)

                for t, s in enumerate(range(sInd, len(odomQ))):
                    qLoc = read_local_raw(query, tstampsQ[s])
                    qGlb = query_global[s]
                    # usually at t=0 there is a meas. update with no motion
                    # separate initialization performed
                    if t == 0:
                        loc.init(odomQ[s], qGlb, qLoc)
                    else:
                        # update state estimate
                        loc.update(odomQ[s-1], query_global[s], qLoc)
                    # check convergence
                    ind_max, converged, score = loc.converged(qGlb, qLoc)
                    # evaluation against ground truth
                    t_err, R_err = pose_err(xyzrpyQ[sInd+t],
                                            refMap.gt_poses[ind_max],
                                            degrees=True)
                    results_tup = (dist_from_start[s] - dist_from_start[sInd],
                                   score, t_err, R_err)
                    trial_results.extend(results_tup)
                    if converged:
                        # identify longest steps to localization for writing
                        # headers in results file
                        if t > t_max:
                            t_max = t
                        break

                # if final trial did not converge, discard from evaluation
                if not converged:
                    break
                trial_no = i + 1
                results.append([trial_no, ntrials, *trial_results])

            # create new folder to store results

            rpath = path.join(RESULTS_DIR, "online")
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
                header = ["trial", "ntrials",
                          *(f"dist_from_start_{t},score_{t},t_err_{t},R_err_{t}"
                            for t in range(t_max+1))]
                csv_file.write(",".join(header) + "\n")
                for result in results:
                    writer.writerow(result)

            # save parameters into results folder for records

            with open(path.join(results_path, 'params.yaml'), 'w') as f:
                yaml.dump(params, f)

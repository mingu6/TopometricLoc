import argparse
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm

import pandas as pd
import yaml

from data.utils import load_pose_data, read_global, read_local_raw
from mapping import RefMap
from settings import DATA_DIR, RESULTS_DIR
from utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run localization experiments for our method or comparisons"))
    parser.add_argument("-d", "--description", type=str, default="",
                        help="description of model for experiment")
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverse", type=str, default="night",
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str, default='t_1_w_10_wd_3.pickle',
                    help="filename containing subsampled reference traverse poses")
    parser.add_argument("-qf", "--query-filename", type=str, default='t_1_w_10.csv',
                    help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-t", "--trials", type=int, default=1000,
                        help="number of trials to run, evenly distributed")
    parser.add_argument("-m", "--method", default="ours",
                        choices=["ours", "RA-L", "3DV", "baseline"])
    args = parser.parse_args()

    ref_traverse = args.reference_traverse
    q_traverse = args.query_traverse
    r_fname = args.reference_filename
    q_fname = args.query_filename
    method = args.method

    # import localization objects with update steps
    if method == "ours":
        from localization import Localization
        if not args.params:
            params_file = 'ours.yaml'
    elif method == "baseline":
        from baselines.geom_verif.base import Localization
        if not args.params:
            params_file = 'baseline.yaml'
    elif method == "RA-L":
        from baselines.RAL20_Topo.base import Localization
        if not args.params:
            params_file = 'RA-L.yaml'
    if args.params:
        params_file = args.params


    # create description of experiment if not specified
    if not args.description:
        description = \
            f"{ref_traverse}_{r_fname[:-7]}_{q_traverse}_{q_fname[:-4]}_{method}"
    else:
        description = args.description
    # create new folder to store results
    rpath = path.join(RESULTS_DIR)
    os.makedirs(rpath, exist_ok=True)  # create base results folder if required
    trials = [int(p.split("_")[-1]) for p in os.listdir(rpath)
              if p.startswith(description)]
    if len(trials) > 0:
        trial_count = max(trials) + 1
        results_path = path.join(rpath, f"{description}_{trial_count}")
    else:
        results_path = path.join(rpath, f"{description}_1")
    os.makedirs(results_path)

    # read in parameters
    params_path = path.abspath(path.join(self_dirpath, "params"))
    with open(path.join(params_path, params_file), 'r') as f:
        params = yaml.safe_load(f)
    # save parameters into results folder for records
    with open(path.join(results_path, 'params.yaml'), 'w') as f:
        yaml.dump(params, f)

    # load map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, r_fname)
    with open(fpath, "rb") as f:
        refMap = pickle.load(f)

    # load query sequence
    tstampsQ, xyzrpyQ, odomQ = load_pose_data(q_traverse, q_fname)
    query_global = read_global(q_traverse, tstampsQ)
    # compute distance travelled (meters) from starting location
    relpose = xyzrpyQ[1:, :2] - xyzrpyQ[:-1, :2]
    relpose = np.vstack((np.zeros(2), relpose))
    dist_from_start = np.cumsum(np.linalg.norm(relpose, axis=1))

    # subsample query traverse for starting point
    spacing = int(len(tstampsQ) / args.trials)  # gap between start of next trial
    start_inds = np.arange(0, len(tstampsQ) - 50, spacing)  # last trial has space

    results = []

    for i, sInd in enumerate(tqdm(start_inds)):
        # setup localization object
        loc = Localization(params, refMap)

        for t, s in enumerate(range(sInd, len(odomQ))):
            qLoc = read_local_raw(q_traverse, tstampsQ[s])
            # usually at t=0 there is a meas. update with no motion
            # separate initialization performed
            if t == 0:
                loc.init(odomQ[s], query_global[s], qLoc)
            else:
                # update state estimate
                loc.update(odomQ[s-1], query_global[s], qLoc)
            # check convergence
            ind_max, converged, _ = loc.converged(
                params['other']['convergence_score'],
                params['other']['convergence_window'])
            if converged:
                break

        # if final trial did not converge, discard from evaluation
        if not converged:
            break
        # evaluation against ground truth
        t_err, R_err = pose_err(xyzrpyQ[sInd+t], refMap.gt_poses[ind_max])
        results.append((i+1, args.trials, dist_from_start[s],
                        dist_from_start[s] - dist_from_start[sInd],
                        t_err, R_err))
    # save results to csv
    df = pd.DataFrame(results, columns=['trial', 'ntrials', 'dist_from_start',
                                        'dist_to_converge', 't_err', 'R_err'])
    df = df.round(1)
    df.to_csv(path.join(results_path, "results.csv"), index=False)

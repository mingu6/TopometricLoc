import argparse
import importlib
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm

import yaml

from .data.utils import load_pose_data, read_global, gt_on_map_status
from .data.reference_maps import RefMap
from . import utils as tutils


def trial_start_locations(n_query, n_steps, n_trials):
    max_start_ind = n_query - n_steps - 1  # ensures all trials will be n_steps
    #start_inds = np.linspace(0, max_start_ind, n_trials).astype(int)
    rng = np.random.RandomState(seed=1)
    start_inds = rng.choice(np.arange(max_start_ind - n_steps - 1), replace=False, size=n_trials)
    return start_inds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run global localization experiments for our method or comparisons"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverses", type=str, nargs="+", required=True,
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str,
                        default='xy_0_t_5.csv',
                        help="filename containing reference map object")
    parser.add_argument("-qf", "--query-filename", type=str, default='xy_3_t_10.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-d", "--descriptor", type=str, default="hfnet",
                        help="global descriptor to evaluate on", choices=["hfnet", "netvlad"])
    parser.add_argument("-w", "--width", type=int, default=10,
                    help="maximum distance for possible transition between nodes")
    parser.add_argument("-m", "--methods", nargs="+", type=str,
                        default=["xu20topo", "stenborg20", "topometric", "nooff"])
    parser.add_argument("-n", "--num-trials", type=int, default=500,
                        help="number of trials to run, evenly distributed")
    parser.add_argument("-di", "--max-dist", type=float, default=90,
                        help="maximum distance travelled (m) per trial")
    args = parser.parse_args()

    n_steps = int(args.max_dist / float(args.query_filename.split('_')[1]))

    # build reference map
    ref_map = RefMap(args.reference_traverse, args.reference_filename, args.descriptor, width=args.width)

    pbarq = tqdm(args.query_traverses, leave=False)
    for query in pbarq:
        pbarq.set_description(query)

        query_tstamps, query_gt, odom_mu, odom_sigma = load_pose_data(query, args.query_filename)
        query_global = read_global(query, query_tstamps, args.descriptor)
        gt_on_map_mask = gt_on_map_status(ref_map.gt_poses, query_gt)
        start_indices = trial_start_locations(len(query_tstamps), n_steps, args.num_trials)

        pbarm = tqdm(args.methods, leave=False)
        for method in pbarm:
            pbarm.set_description(method)

            params = tutils.load_params(method, fname=args.params)
            if method == 'nooff':
                params['other']['off_state'] = False
            Localization = tutils.import_localization("topometric" if method == "nooff" else method)
            localization = Localization(params, ref_map)

            scores = np.empty((args.num_trials, n_steps))
            xy_errs = np.empty((args.num_trials, n_steps))
            rot_errs = np.empty((args.num_trials, n_steps))
            trials_on_map = np.empty((args.num_trials, n_steps), dtype=bool)

            for trial, start_ind in enumerate(tqdm(start_indices, desc="trials", leave=False)):
                # setup localization object
                loc = Localization(params, ref_map)

                for t in range(n_steps):
                    s = start_ind + t
                    if t == 0:
                        loc.init(query_global[s])
                    else:
                        loc.update(odom_mu[s], odom_sigma[s], query_global[s])
                    pose_est, scores[trial, t] = loc.converged()
                    xy_errs[trial, t], rot_errs[trial, t] = tutils.pose_err(query_gt[s], pose_est, degrees=True)
                    trials_on_map[trial, t] = gt_on_map_mask[s]

            results_dir = tutils.create_results_directory(args, query, method, "wakeup")
            np.savez(path.join(results_dir, "results.npz"),
                     convergence_scores=scores,
                     gt_errs_xy=xy_errs,
                     gt_errs_rot=rot_errs,
                     gt_on_map_mask=trials_on_map)
            with open(path.join(results_dir, 'params.yaml'), 'w') as f:
                yaml.dump(params, f)

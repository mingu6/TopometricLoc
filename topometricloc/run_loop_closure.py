import argparse
import importlib
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm, trange

import yaml

from .data.utils import load_pose_data, read_global, gt_on_map_status
from .data.reference_maps import RefMap
from . import utils as utils


def import_base(method_name):
    if method_name in ["topometric", "xu20topo", "stenborg20", "nooff"]:
        method = 'discrete_filters'
    else:
        method = method_name
    base = importlib.import_module(f".loop_closure_detection._lc_{method}", package='topometricloc')
    return base


def save_results(args, query, method, scores, xy_errs, rot_errs, gt_on_map_mask):
    results_dir = utils.create_results_directory(args, query, method, "loop_closure")

    def save_result(fname, score, xy_err, rot_err):
        np.savez(path.join(results_dir, fname),
                 convergence_scores=score,
                 gt_errs_xy=xy_err,
                 gt_errs_rot=rot_err,
                 gt_on_map_mask=gt_on_map_mask)
        return None

    if method in ["topometric", "xu20topo", "stenborg20", "nooff"]:
        scores_fw, scores_bw = scores
        xy_errs_fw, xy_errs_bw = xy_errs
        rot_errs_fw, rot_errs_bw = rot_errs
        save_result("results_fw", scores_fw, xy_errs_fw, rot_errs_fw)
        save_result("results_bw", scores_bw, xy_errs_bw, rot_errs_bw)
    elif method == "baseline":
        save_result("results", scores, xy_errs, rot_errs)
    elif method == "xumcl":
        for i, result in enumerate(results):
            save_result(f"results_{i}", scores, xy_errs, rot_errs)
    else:
        print(f"Unknown method {method}, exiting...")
        raise ValueError
    with open(path.join(results_dir, 'params.yaml'), 'w') as f:
        yaml.dump(params, f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run loop closure experiments for our method or comparisons"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverses", type=str, nargs="+", required=True,
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str,
                        default='xy_0_t_5.csv',
                        help="filename containing reference map object")
    parser.add_argument("-qf", "--query-filename", type=str, default='xy_3_t_10.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="", help="filename containing model parameters")
    parser.add_argument("-d", "--descriptor", type=str, default="hfnet",
                        help="global descriptor to evaluate on", choices=["hfnet", "netvlad", "netvlad_prior", "netvlad_fullres_g"])
    parser.add_argument("-w", "--width", type=int, default=10,
                    help="maximum distance for possible transition between nodes")
    parser.add_argument("-m", "--methods", nargs="+", type=str,
                        default=["xu20topo", "stenborg20", "topometric", "nooff", "baseline", "xu20mcl"])
    args = parser.parse_args()

    ref_map = RefMap(args.reference_traverse, args.reference_filename, args.descriptor, width=args.width)

    pbarq = tqdm(args.query_traverses, leave=False)
    for query in pbarq:
        pbarq.set_description(query)

        query_tstamps, query_gt, odom_mu, odom_sigma = load_pose_data(query, args.query_filename)
        query_global = read_global(query, query_tstamps, args.descriptor)
        gt_on_map_mask = gt_on_map_status(ref_map.gt_poses, query_gt)

        pbarm = tqdm(args.methods, leave=False)
        for method in pbarm:
            pbarm.set_description(method)

            params = utils.load_params(method, fname=args.params)
            if method == 'nooff':
                params['other']['off_state'] = False
            Localization = utils.import_localization("topometric" if method == "nooff" else method)
            localization = Localization(params, ref_map)

            base = import_base(method)
            scores, proposals = base.loop_closure_detection(localization, ref_map, query_global,
                                                            odom_mu, odom_sigma)
            xy_errs, rot_errs = base.evaluate_proposal_error(proposals, query_gt)
            save_results(args, query, method, scores,
                         xy_errs, rot_errs, gt_on_map_mask)

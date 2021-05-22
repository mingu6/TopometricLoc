import argparse
import importlib
import numpy as np
import os
import os.path as path
import pickle
from tqdm import tqdm, trange

import scipy.sparse as sparse
from scipy.special import logsumexp
import yaml

from .data.utils import load_pose_data, read_global, gt_on_map_status
from .data.reference_maps import RefMap
from . import utils as tutils


def forward_update(logalpha, log_trans_mat, log_lhood, data_bc_inds=None):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    log_motion_update = tutils.sparse_nz_sum(log_trans_mat.tocsr(), logalpha, vec_bc_ind=data_bc_inds)
    log_motion_update = tutils.logsumexp_nonzero(log_motion_update, axis=0)
    log_alpha_update = log_motion_update + log_lhood
    return log_alpha_update


def backward_update(logbeta, log_trans_mat, log_lhood, data_bc_inds=None):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    log_update = tutils.sparse_nz_sum(log_trans_mat.tocsc(), logbeta + log_lhood, vec_bc_ind=data_bc_inds)
    log_beta_update = tutils.logsumexp_nonzero(log_update, axis=1)
    return log_beta_update


def forward_recursion(localization, ref_map, query_descriptors, query_odom):
    n_q = query_descriptors.shape[0]
    odom_mu, odom_sigma = query_odom
    # sparse matrix indices for update step in fw/backward
    data_bc_inds = None
    # store measurement lhoods and transition matrices for b/w recursion
    log_meas_lhoods = np.empty((n_q, localization.belief.shape[0]))
    log_transition_probs = []
    # forward recursion joint lhood
    log_alpha = np.empty((n_q, localization.belief.shape[0]))
    for t in trange(n_q, desc='forward', leave=False):
        if t == 0:
            prior = localization.belief.copy()
            meas_lhood = np.log(localization.init(query_descriptors[t]))
            log_alpha[t, :] = np.log(prior) + meas_lhood
        else:
            motion_model = localization._update_motion(odom_mu[t], odom_sigma[t]).copy()
            motion_model.data = np.log(motion_model.data)
            if t == 1:
                data_bc_inds = tutils.bc_vec_to_data_inds(motion_model.tocsr())
            meas_lhood = np.log(localization._update_meas(query_descriptors[t]))
            log_transition_probs.append(motion_model)
            log_alpha[t, :] = forward_update(log_alpha[t-1, :], motion_model, meas_lhood, data_bc_inds=data_bc_inds)
        log_meas_lhoods[t, :] = meas_lhood
    return log_alpha, log_transition_probs, log_meas_lhoods


def backward_recursion(log_trans_mats, log_meas_lhoods):
    n_q, n_r = log_meas_lhoods.shape
    # sparse matrix indices for update step in fw/backward
    data_bc_inds = tutils.bc_vec_to_data_inds(log_trans_mats[0].tocsc())
    # backward recursion joint lhood
    log_beta = np.empty((n_q, n_r))
    log_beta[-1, :] = 0.
    for t in tqdm(reversed(range(n_q-1)), desc='backward', total=n_q-1, leave=False):
        log_beta[t] = backward_update(log_beta[t+1], log_trans_mats[t], log_meas_lhoods[t+1], data_bc_inds=data_bc_inds)
    return log_beta


def process_results(beliefs, localization, query_gt):
    assert beliefs.shape[0] == query_gt.shape[0]
    n_q = beliefs.shape[0]

    convergence_scores = np.empty(n_q, dtype=float)
    gt_errs_xy = np.empty(n_q, dtype=float)
    gt_errs_rot = np.empty(n_q, dtype=float)

    for t in range(n_q):
        pose_est, score = localization.converged(belief=beliefs[t])
        gt_errs_xy[t], gt_errs_rot[t] = tutils.pose_err(query_gt[t], pose_est, degrees=True)
        convergence_scores[t] = score
    return convergence_scores, gt_errs_xy, gt_errs_rot


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
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-d", "--descriptor", type=str, default="hfnet",
                        help="global descriptor to evaluate on", choices=["hfnet", "netvlad"])
    parser.add_argument("-w", "--width", type=int, default=4,
                    help="maximum distance for possible transition between nodes")
    parser.add_argument("-m", "--methods", nargs="+", type=str,
                        default=["xu20topo", "stenborg20", "topometric", "nooff"])
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

            params = tutils.load_params(method, fname=args.params)
            if method == 'nooff':
                params['other']['off_state'] = False
            Localization = tutils.import_localization("topometric" if method == "nooff" else method)
            localization = Localization(params, ref_map)

            log_alpha, log_trans, log_meas = forward_recursion(localization, ref_map, query_global, (odom_mu, odom_sigma))
            log_beta = backward_recursion(log_trans, log_meas)
            log_marginal = logsumexp(log_alpha[-1, :])
            beliefs_forward = np.exp(log_alpha - logsumexp(log_alpha, axis=1)[:, None])
            beliefs_backward = np.exp(log_alpha + log_beta - log_marginal)
            beliefs_backward /= beliefs_backward.sum(axis=1)[:, None]

            scores_fw, xy_errs_fw, rot_errs_fw = process_results(beliefs_forward, localization, query_gt)
            scores_bw, xy_errs_bw, rot_errs_bw = process_results(beliefs_backward, localization, query_gt)

            results_dir = tutils.create_results_directory(args, query, method, "loop_closure")
            np.savez(path.join(results_dir, "results_fw.npz"),
                     convergence_scores=scores_fw,
                     gt_errs_xy=xy_errs_fw,
                     gt_errs_rot=rot_errs_fw,
                     gt_on_map_mask=gt_on_map_mask)
            np.savez(path.join(results_dir, "results_bw.npz"),
                     convergence_scores=scores_bw,
                     gt_errs_xy=xy_errs_bw,
                     gt_errs_rot=rot_errs_bw,
                     gt_on_map_mask=gt_on_map_mask)
            with open(path.join(results_dir, 'params.yaml'), 'w') as f:
                yaml.dump(params, f)

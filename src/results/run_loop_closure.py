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
from settings import DATA_DIR, RESULTS_DIR
from utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))

# off-map thresholds, within this tolerance to be considered on-map
on_xy_thres = 5.
on_rot_thres = 30.


def fw_update(logalpha, log_trans_mat, log_lhood):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    result = np.empty(log_trans_mat.shape[0])
    # ensure csc format so indices are extracted properly
    mat = log_trans_mat.tocsc()
    indptr = mat.indptr
    for i in range(mat.shape[0]):
        indptr = mat.indptr
        indices = mat.indices[indptr[i]:indptr[i+1]]
        # extract relevant column vector entries
        col = logalpha[indices]
        result[i] = logsumexp(col + mat.data[indptr[i]:indptr[i+1]])
    return result + log_lhood


def bw_update(logbeta, log_trans_mat, log_lhood):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    result = np.empty(log_trans_mat.shape[1])
    # ensure csr format so indices are extracted properly
    mat = log_trans_mat.tocsr()
    indptr = mat.indptr
    for i in range(mat.shape[1]):
        indptr = mat.indptr
        indices = mat.indices[indptr[i]:indptr[i+1]]
        # extract relevant column vector entries
        beta_row = logbeta[indices]
        lhood_row = log_lhood[indices]
        result[i] = logsumexp(beta_row + lhood_row +
                              mat.data[indptr[i]:indptr[i+1]])
    return result


def forward_backward(log_prior, log_trans_mats, log_lhoods):
    T = len(log_lhoods)
    N = len(log_prior)
    # forward recursion
    logalpha = np.empty((T, len(prior)))
    for t in tqdm(range(T), desc='fw step'):
        if t == 0:
            logalpha[t, :] = log_prior + log_lhoods[t]
        else:
            logalpha[t, :] = fw_update(logalpha[t-1], log_trans_mats[t-1],
                                       log_lhoods[t])
    log_marginal = logsumexp(logalpha[-1])  # p(X) marginal data lhood
    # backward recursion
    logbeta = np.zeros((T, len(prior)))
    for t in tqdm(reversed(range(T-1)), desc='bw step', total=T-1):
        logbeta[t] = bw_update(logbeta[t+1], log_trans_mats[t], log_lhoods[t+1])
    # forward beliefs
    fw_beliefs = np.empty((T, N))
    for t in range(T):
        fw_beliefs[t] = np.exp(logalpha[t] - logsumexp(logalpha[t]))
    # backward beliefs
    bw_beliefs = np.empty((T, N))
    bw_beliefs = np.exp(logalpha + logbeta - log_marginal)
    return fw_beliefs, bw_beliefs


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
    parser.add_argument("-qf", "--query-filename", type=str, default='xy_2_t_15.csv',
                        help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    parser.add_argument("-m", "--methods", nargs="+", type=str,
                        choices=["ours", "xu20", "stenborg20", "baseline", "noverif", "nooff"],
                        default=["ours", "xu20", "stenborg20", "noverif", "nooff"])
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

        pbarm = tqdm(args.methods)
        for method in pbarm:
            pbarm.set_description(method)

            # for each method (ours/comparisons), import assoc. module

            if method == "ours":
                from ours.localization import LocalizationFull as Localization
                from ours.localization import convergedFull as converged
            elif method == "noverif":
                from ours.localization import LocalizationNoVerif as Localization
                from ours.localization import convergedNoVerif as converged
            elif method == "nooff":
                from ours.localization import LocalizationNoOff as Localization
                from ours.localization import convergedNoOff as converged
            else:
                localization = importlib.import_module(
                    f"comparison_methods.{method}.localization")
                Localization = localization.Localization
                from ours.localization import convergedNoOff as converged

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
            scores = []
            checks = []
            xy_errs = []
            rot_errs = []
            ref_inds = []
            on_map_stats = []
            off_probs = []

            # save intermediate
            meas_lhoods = []
            transition_matrices = []
            prior = None

            # setup localization object
            loc = Localization(params, refMap)
            for i in tqdm(range(len(tstampsQ)), desc="extract", leave=False):

                qLoc = read_local_raw(query, tstampsQ[i])
                qGlb = query_global[i]
                qmu, qSigma = muQ[i], SigmaQ[i]
                # usually at t=0 there is a meas. update with no motion
                # separate initialization performed
                if i == 0:
                    prior = np.log(loc.belief)
                    lhood = loc.init(qmu, qSigma, qGlb, qLoc)
                    meas_lhoods.append(np.log(lhood))
                else:
                    # update state estimate
                    trans_mat = loc._update_motion(qmu, qSigma)
                    lhood = loc._update_meas(qGlb, qLoc)
                    trans_mat.data = np.log(trans_mat.data)
                    meas_lhoods.append(np.log(lhood))
                    transition_matrices.append(trans_mat)

            fw_beliefs, bw_beliefs = forward_backward(
                prior, transition_matrices, meas_lhoods)

            # check convergence of forward beliefs

            T = len(fw_beliefs)

            scores_fw = np.empty(T, dtype=float)
            checks_fw = np.empty(T, dtype=bool)
            ref_inds_fw = np.empty(T, dtype=int)
            xy_err_fw = np.empty(T, dtype=float)
            rot_err_fw = np.empty(T, dtype=float)
            off_prob_fw = np.empty(T, dtype=float)

            scores_bw = np.empty(T, dtype=float)
            checks_bw = np.empty(T, dtype=bool)
            ref_inds_bw = np.empty(T, dtype=int)
            xy_err_bw = np.empty(T, dtype=float)
            rot_err_bw = np.empty(T, dtype=float)
            off_prob_bw = np.empty(T, dtype=float)

            for t in tqdm(range(T), desc='converge', leave=False):
                qLoc = read_local_raw(query, tstampsQ[i])
                # check convergence save forward pass stats
                ind_prop, check, score = converged(
                    loc, query_global[t], qLoc, fw_beliefs[t])
                xy_err, rot_err = pose_err(gtQ[t], refMap.gt_poses[ind_prop],
                                           degrees=True)
                scores_fw[t] = score
                checks_fw[t] = check
                ref_inds_fw[t] = ind_prop
                xy_err_fw[t] = xy_err
                rot_err_fw[t] = rot_err
                off_prob = fw_beliefs[t, -1] if method in ["ours", "noverif"] else 0.
                off_prob_fw[t] = off_prob
                # check convergence save backward pass stats
                ind_prop, check, score = converged(
                    loc, query_global[t], qLoc, bw_beliefs[t])
                xy_err, rot_err = pose_err(gtQ[t], refMap.gt_poses[ind_prop],
                                           degrees=True)
                scores_bw[t] = score
                checks_bw[t] = check
                ref_inds_bw[t] = ind_prop
                xy_err_bw[t] = xy_err
                rot_err_bw[t] = rot_err
                off_prob = bw_beliefs[t, -1] if method in ["ours", "noverif"] else 0.
                off_prob_bw[t] = off_prob

            results = {"forward": {"scores": scores_fw,
                                   "checks": checks_fw,
                                   "ref_inds": ref_inds_fw,
                                   "xy_err": xy_err_fw,
                                   "rot_err": rot_err_fw,
                                   "off_probs": off_prob_fw,
                                   "on_status": q_on_map},
                       "backward": {"scores": scores_bw,
                                    "checks": checks_bw,
                                    "ref_inds": ref_inds_bw,
                                    "xy_err": xy_err_bw,
                                    "rot_err": rot_err_bw,
                                    "off_probs": off_prob_bw,
                                    "on_status": q_on_map}}

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

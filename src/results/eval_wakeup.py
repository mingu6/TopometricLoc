import argparse
import os
import os.path as path
import matplotlib.pyplot as plt
import numpy as np

import pickle
import pandas as pd

from settings import RESULTS_DIR

self_dirpath = os.path.dirname(os.path.abspath(__file__))

colors = {"Ours": "green",
          "No Verif": "yellow",
          "No Off": "purple",
          "Baseline": "blue",
          "Xu20": "red",
          "Stenborg20": "orange"}


linestyle = ["dashed", "solid", "dashdot"]

# convergence scores, more granular near 1
scores_vec = np.hstack((np.linspace(0., 0.9, 45, endpoint=False),
                        np.linspace(0.9, 1., 100, endpoint=False)))


def load_results(fname):
    """
    Load generated results from wakeup trials. Returns None if no file.
    """
    fpath = path.join(RESULTS_DIR, 'wakeup', fname, 'results.pickle')
    try:
        with open(fpath, 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError as e:
        print(e)
        results = None
    return results


def localize_at_thres(thres, scores, checks, off_infer):
    """
    Given set of check flags and score data, locate first step where
    proposal is accepted (score > thres and check). Assumes threshold
    is a vector containing multiple thresholds

    score above threshold and pass checks -> localize
    predict off-map -> localize
    wait until end do nothing failure unless actually off-map
    predict off but not really failure

    only output indices in this function
    """
    above_thres = scores[None, :] >= thres[:, None]
    above_thres = np.logical_and(above_thres, checks[None, :])
    off_infer_bool = np.full(len(scores), False, dtype=bool)
    if off_infer != -2:
        off_infer_bool[off_infer] = True
    localized = np.logical_or(above_thres, off_infer_bool[None, :])
    # argmax returns first index or 0 if all failed
    indices = np.argmax(localized, axis=1)
    # failed to localize, conditions never met
    failure = np.logical_not(np.any(localized, axis=1))
    if np.any(failure):
        indices[failure] = -1
    return indices


def label_success(indices, off_pred, dists, xy_errs, rot_errs, on_maps,
                  xy_thres, rot_thres):
    """
    Given localization error and on-map status of query, label each trial
    success or failure. Takes indices of proposals pre-computed in
    localize_at_thres function.

    score above threshold and pass checks -> localize
    predict off-map -> localize
    wait until end do nothing failure unless actually off-map
    predict off but not really failure

    only assess success or not
    """
    # successful pose error
    xy_success = xy_errs[indices] < xy_thres
    rot_success = rot_errs[indices] < rot_thres
    success = np.logical_and(xy_success, rot_success)
    success = np.logical_and(success, indices != -1)
    # must be on map for pose error success
    success = np.logical_and(success, indices != off_pred)
    # failure when localization fails to occur, unless localization fails
    # and sequence ends on a query that is actually off-map
    off_success_end = np.logical_and(indices == -1, not on_maps[-1])
    success = np.logical_or(success, off_success_end)
    # infer off only for our method, predict off-map
    off_correct = np.logical_not(on_maps)[off_pred]
    if off_correct:
        off_pred = indices == off_pred
    else:
        off_pred = np.logical_and(success, False)  # all False
    success = np.logical_or(success, off_pred)
    return success, dists[indices]


def infer_off_wakeup(off_probs):
    # conditions
    off_thres = 0.70
    duration = 12
    # identify contiguous segments of off-map predictions
    num_off = 0
    ind_off = -2
    for i, prob in enumerate(off_probs):
        if prob > off_thres:
            num_off += 1
        else:
            num_off = 0
        if num_off >= duration:
            ind_off = i
            break
    return ind_off


def succ_by_dist_curve(success, dist):
    success_prop = success.mean(axis=0)
    dist_mean = dist.mean(axis=0)
    #dist_mean = np.median(dist, axis=0)
    return dist_mean, success_prop


def propn_loc_at_dist_curve(score_ind, successes, dists, loc_indices):
    dist_uniform = np.linspace(0., dists.max(), 100)
    # at optimal score, look at success rate and localized status
    success_at_ind = successes[:, score_ind]
    dists_at_ind = dists[:, score_ind]
    loc_ind_at_ind = loc_indices[:, score_ind]
    # look at proportion localized by each distance
    localized_by = dists_at_ind[:, None] < dist_uniform[None, :]
    localized_by = np.logical_and(localized_by, (loc_ind_at_ind != -1)[:, None])
    success_by = np.logical_and(localized_by, success_at_ind[:, None])
    propn_success_by = success_by.mean(axis=0)
    return dist_uniform, propn_success_by


def main(args):
    xy_thres = args.trans_err
    rot_thres = args.rot_err
    # read file with result filenames to read
    df_desc = pd.read_csv(path.join(self_dirpath, args.filename))
    df_desc_rows = df_desc.values.tolist()
    # keep record of curve data
    success_curves = {}
    propn_curves = {}
    for traverse, method, fname in df_desc_rows:
        results = load_results(fname)
        if results is not None:
            if traverse not in success_curves:
                success_curves[traverse] = {}
                success_curves[traverse][method] = []
            elif method not in success_curves[traverse]:
                success_curves[traverse][method] = []
            if traverse not in propn_curves:
                propn_curves[traverse] = {}
                propn_curves[traverse][method] = []
            elif method not in propn_curves[traverse]:
                propn_curves[traverse][method] = []
            for xy_t, r_t in zip(xy_thres, rot_thres):
                # success and distance travelled for thresholds. Raw input for curves
                dist_results = []
                success_results = []
                loc_inds = []
                for i, res in enumerate(results):
                    # extract result data for trial
                    dist = res['dist']  # distance travelled from init
                    xy_err = res['xy_err']  # transl. err (m)
                    rot_err = res['rot_err']  # orient. err. (deg)
                    scores = res['scores']  # convergence score in [0, 1]
                    checks = res['checks']  # flags, requires True to localize
                    on_maps = res['on_status']
                    off_probs = res['off_probs']
                    off_infer_ind = infer_off_wakeup(off_probs)
                    # localize at required thresholds, find indices
                    loc_indices = localize_at_thres(scores_vec, scores, checks,
                                                    off_infer_ind)
                    successes, dists = label_success(loc_indices, off_infer_ind, dist,
                                                     xy_err, rot_err, on_maps,
                                                     xy_t, r_t)
                    # distance travelled and success for trial at all thresholds
                    dist_results.append(dists)
                    success_results.append(successes)
                    loc_inds.append(loc_indices)
                # generate actual curve values
                dist_results = np.asarray(dist_results)
                success_results = np.asarray(success_results)
                loc_inds = np.asarray(loc_inds)
                x, y = succ_by_dist_curve(success_results, dist_results)
                success_curves[traverse][method].append((x, y))
                # proportion localized @ dist curve
                score_ind_at_best = np.argmax(y)  # highest success
                if np.any(y > 0.99):
                    score_ind_at_best = min(score_ind_at_best, np.argmax(y > 0.99))
                dist_by, success_by = propn_loc_at_dist_curve(
                    score_ind_at_best, success_results, dist_results, loc_inds)
                propn_curves[traverse][method].append((dist_by, success_by))

    # plot curves
    fig, axs = plt.subplots(1, len(success_curves.keys()))
    fig.suptitle("Propotion success by mean distance to localize (m)",
                 fontsize=24)
    for i, (traverse, save) in enumerate(success_curves.items()):
        for j, (method, curves) in enumerate(save.items()):
            if method != "Baseline":
                for k, curve in enumerate(curves):
                    axs[i].plot(curve[0], curve[1], color=colors[method],
                                linestyle=linestyle[k],
                                label=f"{method} @ {xy_thres[k]}m, {rot_thres[k]} deg")
            else:
                axs[i].scatter([curve[0][0]], [curve[1][0]], color=colors[method],
                               s=50, label=f"{method} @ {xy_thres[k]}m, {rot_thres[k]} deg")
        axs[i].set_title(f"{traverse}", fontsize=16)
        axs[i].set_aspect(0.8/axs[i].get_data_ratio(), adjustable='box')
        axs[i].set_xlabel("Mean distance travelled (m)", fontsize=16)
        axs[i].set_ylabel("Proportion of successful trials", fontsize=16)
    axs[-1].legend()
    old_fig_size = fig.get_size_inches()
    fig.set_size_inches(old_fig_size[0] * 2.0, old_fig_size[1] * 2.0)
    fig.tight_layout()

    # plot curves
    fig1, axs1 = plt.subplots(1, len(propn_curves.keys()))
    fig1.suptitle("Proportion successfully localized by distance",
                  fontsize=24)
    for i, (traverse, save) in enumerate(propn_curves.items()):
        for j, (method, curves) in enumerate(save.items()):
            for k, curve in enumerate(curves):
                axs1[i].plot(curve[0], curve[1], color=colors[method],
                            linestyle=linestyle[k],
                            label=f"{method} @ {xy_thres[k]}m, {rot_thres[k]} deg")
        axs1[i].set_title(f"{traverse}", fontsize=16)
        axs1[i].set_aspect(0.8/axs1[i].get_data_ratio(), adjustable='box')
        axs1[i].set_xlabel("Distance travelled (m)", fontsize=16)
        axs1[i].set_ylabel("Proportion of trials successfully localized", fontsize=16)
    axs1[-1].legend()
    old_fig_size = fig1.get_size_inches()
    fig1.set_size_inches(old_fig_size[0] * 2.0, old_fig_size[1] * 2.0)
    fig1.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Aggregate results for comparison tables"))
    parser.add_argument("-f", "--filename", type=str, default='wakeup_fnames.csv',
                    help="filename containing result descriptions to aggregate")
    parser.add_argument("-te", "--trans-err", type=float, nargs="+",
                        default=[3., 5., 10.], help="error tolerance (m)")
    parser.add_argument("-re", "--rot-err", type=float, nargs="+",
                        default=[15., 30., 180.], help="error tolerance (deg)")
    args = parser.parse_args()

    main(args)

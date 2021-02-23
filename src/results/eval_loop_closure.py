import argparse
import os
import os.path as path
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
import numpy as np

import pickle
import pandas as pd
import seaborn as sns; sns.set(); sns.set_style("white"); sns.set_context(font_scale=1.1)

from settings import RESULTS_DIR

self_dirpath = os.path.dirname(os.path.abspath(__file__))

colors = {"Ours": "green",
          "No Verif": "m",
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
    fpath = path.join(RESULTS_DIR, 'loop_closure', fname, 'results.pickle')
    try:
        with open(fpath, 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError as e:
        print(e)
        results = None
    return results


def localized_at_thres(score_thres, scores, checks):
    localized = scores[:, None] > score_thres[None, :]
    localized = np.logical_and(localized, checks[:, None])
    return localized


def success_at_tol(localized, xy_err, rot_err, on_status, xy_tol, rot_tol):
    # TP if within map and classified within map
    xy_success = xy_err < xy_tol
    rot_success = rot_err < rot_tol
    on_success = np.logical_and(xy_success, rot_success)
    TP = np.logical_and(localized, on_success[:, None]).sum(axis=0)
    # FP if localized wrong
    FP = np.logical_and(localized,
                        np.logical_not(on_success[:, None])).sum(axis=0)
    # TN if off-map and no data assoc.
    TN = np.logical_and(np.logical_not(localized),
                        np.logical_not(on_status[:, None])).sum(axis=0)
    # FN if no assoc. but not off-map
    FN = np.logical_and(np.logical_not(localized),
                        on_status[:, None]).sum(axis=0)
    return TP, FP, TN, FN


def precision_recall(TP, FP, TN, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def precision_recall_curve(score_thres, data, xy_tol, rot_tol):
    localized = localized_at_thres(score_thres, data['scores'], data['checks'])
    TP, FP, TN, FN = success_at_tol(localized, data['xy_err'], data['rot_err'],
                                    data['on_status'], xy_tol, rot_tol)
    precision, recall = precision_recall(TP, FP, TN, FN)
    return precision, recall


def main(args):
    xy_thres = args.trans_err
    rot_thres = args.rot_err
    # read file with result filenames to read
    df_desc = pd.read_csv(path.join(self_dirpath, args.filename))
    df_desc_rows = df_desc.values.tolist()
    # keep record of curve data
    PR_curves = {}
    for traverse, method, fname in df_desc_rows:
        results = load_results(fname)
        if results is not None:
            if traverse not in PR_curves:
                PR_curves[traverse] = {}
                PR_curves[traverse][method] = {}
            elif method not in PR_curves[traverse]:
                PR_curves[traverse][method] = {}
            # load data
            fw_prec, fw_recall = precision_recall_curve(scores_vec,
                                                        results['forward'],
                                                        xy_thres, rot_thres)
            bw_prec, bw_recall = precision_recall_curve(scores_vec,
                                                        results['backward'],
                                                        xy_thres, rot_thres)
            PR_curves[traverse][method]["fw"] = (fw_prec, fw_recall)
            PR_curves[traverse][method]["bw"] = (bw_prec, bw_recall)

    # plot curves
    fig, axs = plt.subplots(1, len(PR_curves.keys()))
    fig.suptitle("Loop Closure Detection: Precision/Recall",
                 fontsize=24)
    for i, (traverse, save) in enumerate(PR_curves.items()):
        for j, (method, curves) in enumerate(save.items()):
            if method != "Baseline":
                if method == 'Ours':
                    zorder = 20
                else:
                    zorder = 5
                for k, (fwbw, curve) in enumerate(curves.items()):
                    axs[i].plot(curve[1], curve[0], color=colors[method],
                                linestyle=linestyle[k],
                                linewidth=3,
                                label=f"{method} {fwbw}", zorder=zorder)
            else:
                axs[i].scatter([curve[1][0]], [curve[0][0]], color=colors[method],
                               s=50, label=f"{method}")
        axs[i].set_title(f"{traverse}", fontsize=16)
        axs[i].set_xlabel("Recall", fontsize=16)
        axs[i].set_ylabel("Precision", fontsize=16)
        axs[i].set_ylim(0.8, 1.)
        axs[i].set_aspect(0.8/axs[i].get_data_ratio(), adjustable='box')
    axs[-1].legend()
    old_fig_size = fig.get_size_inches()
    fig.set_size_inches(old_fig_size[0] * 3.0, old_fig_size[1] * 1.5)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate evaluation plots for loop closure"))
    parser.add_argument("-f", "--filename", type=str, default='wakeup_fnames.csv',
                    help="filename containing result descriptions to aggregate")
    parser.add_argument("-te", "--trans-err", type=float,
                        default=3., help="error tolerance (m)")
    parser.add_argument("-re", "--rot-err", type=float,
                        default=15., help="error tolerance (deg)")
    args = parser.parse_args()

    main(args)

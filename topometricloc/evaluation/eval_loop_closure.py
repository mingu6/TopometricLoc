import argparse
import os
import os.path as path
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)
import numpy as np

import pandas as pd
import seaborn as sns; sns.set(); sns.set_style("white"); sns.set_context(font_scale=1.1)

from . import utils
from ..settings import RESULTS_DIR


def generate_pr_curves(results, xy_thres, rot_thres):
    xy_errs = results['gt_errs_xy']
    rot_errs = results['gt_errs_rot']
    scores = results['convergence_scores']
    gt_on_mask = results['gt_on_map_mask']
    pred_within_tol = utils.check_gt_err_within_tol(xy_errs, rot_errs, xy_thres, rot_thres)
    scores_thres = scores.min() + (scores.max() - scores.min()) * utils.convergence_score_thresh
    converged_mask = scores[None, :] > scores_thres[:, None]
    confusion_status = utils.determine_confusion_status(converged_mask, pred_within_tol, gt_on_mask)
    precision, recall = utils.precision_recall(confusion_status)
    return precision, recall


def load_results(method_name, description):
    results_dir = path.join(RESULTS_DIR, "loop_closure", description)
    if method_name in ["Ours", "No Off", "Stenborg20", "Xu20Topo"]:
        results_fw = utils.load_results(path.join(results_dir, "results_fw.npz"))
        results_bw = utils.load_results(path.join(results_dir, "results_bw.npz"))
        results = (results_fw, results_bw)
    elif method_name == "Baseline":
        results = utils.load_results(path.join(results_dir, "results.npz"))
    elif method_name == "Xu20MCL":
        result_fnames = [fname for fname in os.listdir(results_dir) if fname[:-4] == ".npz"]
        for fname in result_fnames:
            results = utils.load_results(path.join(results_dir, fname))
    else:
        print(f"Unknown method {method_name}, skipping...")
        results = None
    return results


def pr_curves_from_results(description, results, xy_thres, rot_thres):
    if results is not None:
        _, method_name = description.split("\/")
        curve_data = []
        if method_name in ["Ours", "No Off", "Stenborg20", "Xu20Topo"]:
            results_fw, results_bw = results
            if results_fw is not None:
                precision_fw, recall_fw = generate_pr_curves(results_fw, xy_thres, rot_thres)
                curve_data.append((description + ' fw', precision_fw, recall_fw))
            if results_bw is not None:
                precision_bw, recall_bw = generate_pr_curves(results_bw, xy_thres, rot_thres)
                curve_data.append((description + ' bw', precision_bw, recall_bw))
        elif method_name == "Baseline":
            precision, recall = generate_pr_curves(results, xy_thres, rot_thres)
            curve_data.append((description, precision, recall))
        elif method_name == "Xu20MCL":
            precision, recall = generate_pr_curves(results, xy_thres, rot_thres)
            curve_data.append((description, precision, recall))
        else:
            print(f"Unknown method {method_name}, skipping...")
            curve_data = None
    else:
        curve_data = None
    return curve_data


def main(results_fname, xy_thres, rot_thres):
    self_dirpath = os.path.dirname(os.path.abspath(__file__))
    results_file = pd.read_csv(path.join(self_dirpath, 'eval_files', results_fname))

    # load and process raw results data
    curve_data = []
    for _, row in results_file.iterrows():
        description = row['description']
        query_traverse, method_name = row['query'], row['method']
        joined = query_traverse + "\/" + method_name
        results = load_results(method_name, description)
        curves = pr_curves_from_results(joined, results, xy_thres, rot_thres)
        if curves is not None:
            curve_data.extend(curves)

    # display recall at 99% precision as a summary statistic
    for summary, precision, recall in curve_data:
        traverse, method = summary.split('\/')
        if precision.max() >= 0.99:
            max_recall_at_99 = recall[precision >= 0.99].max()
            print(f"Method: {method}, Traverse: {traverse}, recall: {max_recall_at_99:.3f}")

    # determine number of plots by query traverses with valid results data
    query_traverses = [summary.split('\/')[0] for summary, _, _ in curve_data]
    uniq_query_traverses = sorted(list(set(query_traverses)))
    num_query = len(uniq_query_traverses)
    if num_query == 0:
        raise ValueError("No results loaded, exiting...")

    fig, axs = plt.subplots(1, num_query)
    if num_query == 1:
        axs = [axs]  # avoids indexing error if only one query
    axes_dict = dict(zip(uniq_query_traverses, axs))

    for summary, precision, recall in curve_data:
        traverse, method_full = summary.split('\/')
        method = method_full[:-3] if method_full[-3:] in [" fw", " bw"] else method_full
        ax = axes_dict[traverse]
        ls = "dashed" if method_full[-3:] == " fw" else "solid"
        ax.plot(recall, precision, color=utils.colors[method], linewidth=3,
                linestyle=ls, label=f"{method_full}")
        ax.set_title(traverse, fontsize=16)
        ax.set_xlabel("Recall", fontsize=16)
        ax.set_ylabel("Precision", fontsize=16)
        ax.set_ylim(0.8, 1.)
        ax.set_aspect(0.5 / ax.get_data_ratio(), adjustable='box')

    fig.suptitle(f"Loop Closure Detection: Precision/Recall {xy_thres:.0f}m {rot_thres:.0f} deg", fontsize=24)
    axs[-1].legend()
    fig.set_size_inches(fig.get_size_inches()[0] * 3.0, fig.get_size_inches()[1] * 1.0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate evaluation plots for loop closure"))
    parser.add_argument("-f", "--filename", type=str, default='wakeup_fnames.csv',
                    help="filename containing result descriptions to aggregate")
    parser.add_argument("-te", "--trans-err", type=float,
                        default=5., help="error tolerance (m)")
    parser.add_argument("-re", "--rot-err", type=float,
                        default=30., help="error tolerance (deg)")
    args = parser.parse_args()

    main(args.filename, args.trans_err, args.rot_err)

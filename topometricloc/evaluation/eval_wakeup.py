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


def generate_pr_curves(results, xy_thres, rot_thres):
    xy_errs = results['gt_errs_xy']
    rot_errs = results['gt_errs_rot']
    scores = results['convergence_scores']
    n_trials = scores.shape[0]
    # identify which trials converged before end of sequence and if so, if they converged to the right place
    pred_within_tol = utils.check_gt_err_within_tol(xy_errs, rot_errs, xy_thres, rot_thres)
    tstep_converged = utils.tstep_first_converged(utils.convergence_score_thresh, scores)
    converged_mask = tstep_converged != -1
    within_gt_tol = pred_within_tol[np.arange(n_trials), tstep_converged]
    gt_on_mask = results['gt_on_map_mask'][np.arange(n_trials), tstep_converged]
    # build PR curves
    confusion_status = utils.determine_confusion_status(converged_mask, within_gt_tol, gt_on_mask)
    precision, recall = utils.precision_recall(confusion_status)
    return precision, recall


def main(results_fname, xy_thres, rot_thres):
    self_dirpath = os.path.dirname(os.path.abspath(__file__))
    results_file = pd.read_csv(path.join(self_dirpath, 'eval_files', results_fname))

    # load and process raw results data
    curve_data = []
    for _, row in results_file.iterrows():
        description = row['description']
        query_traverse = row['query']
        method_name = row['method']
        results = utils.load_results(description, exper='wakeup')
        if results is not None:
            precision, recall = generate_pr_curves(results, xy_thres, rot_thres)
            curve_data.append(('\/'.join([query_traverse, method_name]), precision, recall))

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
        traverse, method = summary.split('\/')
        ax = axes_dict[traverse]
        ax.plot(recall, precision, color=utils.colors[method], linewidth=3, label=method)
        ax.set_title(traverse, fontsize=16)
        ax.set_xlabel("Recall", fontsize=16)
        ax.set_ylabel("Precision", fontsize=16)
        ax.set_ylim(0.4, 1.)
        ax.set_aspect(0.8 / ax.get_data_ratio(), adjustable='box')

    fig.suptitle(f"Wakeup/Global Localization: Precision/Recall {xy_thres:.0f}m {rot_thres:.0f} deg", fontsize=24)
    axs[-1].legend()
    fig.set_size_inches(fig.get_size_inches()[0] * 3.0, fig.get_size_inches()[1] * 1.0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate evaluation plots for loop closure"))
    parser.add_argument("-f", "--filename", type=str, default='wakeup_fnames.csv',
                    help="filename containing result descriptions to aggregate, stored in same dir as script")
    parser.add_argument("-te", "--trans-err", type=float, default=3., help="error tolerance (m)")
    parser.add_argument("-re", "--rot-err", type=float, default=15., help="error tolerance (deg)")
    args = parser.parse_args()

    main(args.filename, args.trans_err, args.rot_err)

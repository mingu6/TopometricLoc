import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as path

import pandas as pd

from settings import RESULTS_DIR

self_dirpath = os.path.dirname(os.path.abspath(__file__))

colors = {"Ours": "green",
          "Baseline": "blue",
          "Xu20": "red",
          "Stenborg20": "orange"}

linestyle = ["dashed", "solid"]


def preprocess_results(results_path):
    """
    reads raw results from results.csv files and turns processes them
    into a form more usable for plotting and analysis
    Args:
        results_path: path of results.csv file
    returns:
        results: processed results
    """
    results = []
    with open(results_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(reader):
            # skip header
            if i == 0:
                continue
            # recover results from row
            row_results = np.asarray(row[2:], dtype=np.float32).reshape(-1, 4)
            dist_from_start = row_results[:, 0]
            scores = row_results[:, 1]
            t_errs = row_results[:, 2]
            R_errs = row_results[:, 3]
            # save to list
            results.append((dist_from_start, scores, t_errs, R_errs))
    return results


def precision_at_tol(t_errs, R_errs, t_thres, R_thres):
    """
    Compute precision of results given ground truth error tolerances
    Args:
        t_errs: array (Nx1): translational errors for N trials
        R_errs: array (Nx1): orientation errors for N trials
        t_thres: threshold on translational error (m) for trial success
        R_thres: threshold on orientation error (deg) for trial success
    Returns:
        precision: precision [0, 1] for all results
    """
    success = np.logical_and(t_errs < t_thres, R_errs < R_thres)
    return success.sum() / len(success)


def localize_with_score(results, score_thres, baseline=False):
    """
    Given score and full results, localize each trial i.e. take the
    ground truth error at the timestep where the score is higher than
    the provided threshold.
    Args:
        results: processed results in a list outputted by the
                 preprocess_results function
        score_thres: convergence score for localization
    Returns:
        t_errs: translation error (m) at convergence
        R_errs: orientation error (deg) at convergence
        dist_from_starts: distance (m) travelled before convergence
    """
    t_errs = []
    R_errs = []
    dist_from_starts = []
    for dist_from_start, score, t_err, R_err in results:
        if baseline:
            t_errs.append(t_err[-1])
            R_errs.append(R_err[-1])
            dist_from_starts.append(dist_from_start[-1])
        else:
            for t, s in enumerate(score):
                if s > score_thres:
                    t_errs.append(t_err[t])
                    R_errs.append(R_err[t])
                    dist_from_starts.append(dist_from_start[t])
                    break
    t_errs = np.asarray(t_errs)
    R_errs = np.asarray(R_errs)
    dist_from_starts = np.asarray(dist_from_starts)
    return t_errs, R_errs, dist_from_starts


def score_at_prec(results, prec_lvl, t_thres, R_thres):
    """
    Find convergence score for filters given desired level of precision.
    Args:
        results: processed results in a list outputted by the
                 preprocess_results function
        prec_lvl: desired level of precision (e.g. 0.99)
        t_thres: maximum translation error (m) to be deemed successful
        R_thres: maximum orientation error (deg) to be deemed successful
    Returns:
        score (float, [0, 1]): convergence score that yields the
                               desired precision level
    """
    # find unique set of confidence scores, iterate over scores to yield
    # precision levels for each score
    all_scores = [result[1] for result in results]
    # recover maximum convergence score during evaluation run
    score_thres = min(max(sc) for sc in all_scores)
    scores = np.linspace(score_thres, 0., 100)

    prev_score = None

    for score in scores:
        t_errs, R_errs, dist_from_start = localize_with_score(results, score)
        P = precision_at_tol(t_errs, R_errs, t_thres, R_thres)
        # stop when precision level reached, use score that meets precision level
        if P <= prec_lvl and prev_score is not None:
            desired_score = prev_score
            break
        else:
            desired_score = score
        prev_score = score
    return desired_score


def loc_curve_at_prec(results, prec_lvl, t_thres, R_thres, baseline=False):
    """
    Generates curve with y-axis probability of not localized and x-axis
    being the distance of query sequence (m). Measures both localization
    accuracy and distance to convergence in single graph.
    Args:
        results: results read from results.csv file for traverse+method combo
        prec_lvl: desired level of precision e.g. 0.99
        t_thres: maximum translation error (m) to be deemed successful
        R_thres: maximum orientation error (deg) to be deemed successful
    Returns:
        x: curve x value (distance until localized in m)
        y: proportion of trials not localized by x meters
    """
    if not baseline:
        scoreP = score_at_prec(results, prec_lvl, t_thres, R_thres)
    else:
        scoreP = 0.
    _, _, dist_from_start = localize_with_score(results, scoreP,
                                                baseline=baseline)
    x = np.sort(dist_from_start)
    y = np.linspace(0., 1., len(x))  # empirical cdf
    return x, y


def prec_by_median_dist(results, t_thres, R_thres):
    """
    Compute curve where y-axis is precision and x-axis is dist to loc.
    Args:
        results: processed results in a list outputted by the
                 preprocess_results function
        t_thres: maximum translation error (m) to be deemed successful
        R_thres: maximum orientation error (deg) to be deemed successful
    Returns:
        dist_to_loc: x-axis of graph
        prec: y-axis of graph
    """
    # find unique set of confidence scores, iterate over scores to yield
    # precision levels for each score
    all_scores = [result[1] for result in results]
    # recover maximum convergence score during evaluation run
    score_thres = min(max(sc) for sc in all_scores)
    scores = np.linspace(0., score_thres, 100)

    precisions = np.empty(len(scores))
    dist_to_loc = np.empty(len(scores))

    for i, score in enumerate(scores):
        t_errs, R_errs, dist_from_start = localize_with_score(results, score)
        precisions[i] = precision_at_tol(t_errs, R_errs, t_thres, R_thres)
        dist_to_loc[i] = np.median(dist_from_start)
    return dist_to_loc, precisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Aggregate results for comparison tables"))
    parser.add_argument("-f", "--filename", type=str, default='results_online.csv',
                    help="filename containing result descriptions to aggregate")
    parser.add_argument("-t", "--transl-err", type=float, default=5.,
                        help="translational error (m) threshold for success")
    parser.add_argument("-R", "--rot-err", type=float, default=30.,
                        help="rotational error threshold (deg) for success")
    parser.add_argument("-P", "--precision-levels", nargs="+", type=float,
                        default=[0.95, 0.99], help="precision levels to draw curves")
    args = parser.parse_args()

    df_desc = pd.read_csv(path.join(self_dirpath, args.filename))
    df_desc_rows = df_desc.values.tolist()
    # save result filenames for lookup during plotting
    descs = {}
    for query, method, desc in df_desc_rows:
        descs[query+method] = desc
    # extract all methods to be evaluated
    methods = set(row[1] for row in df_desc_rows)
    queries = set(row[0] for row in df_desc_rows)

    # setup figures
    fig, ax = plt.subplots(1, len(queries))

    ############## First figure, propn localized by x meters ############

    for i, query in enumerate(queries):
        for j, prec in enumerate(args.precision_levels):
            for method in methods:
                try:
                    desc = descs[query+method]
                    results = preprocess_results(path.join(RESULTS_DIR, desc,
                                                           'results.csv'))
                    # baseline method has no variable convergence threshold,
                    # so do not plot curve and output summary statistics
                    if method == "Baseline":
                        t_errs, R_errs, dists_from_start = \
                            localize_with_score(results, 0., baseline=True)
                        prec_bl = precision_at_tol(t_errs, R_errs,
                                                   args.transl_err, args.rot_err)
                        if j == 0:
                            print(f"Traverse: {query}, Method: baseline, "
                                  f"Precision: {prec_bl}, Median dist. to loc.:"
                                  f" {np.median(dists_from_start):.1f}m")
                    else:
                        x, y = loc_curve_at_prec(results, prec, args.transl_err,
                                                 args.rot_err)
                        ax[i].plot(x, y, label=f"{method} @ P = {prec}",
                                   color=colors[method], linewidth=3,
                                   linestyle=linestyle[j])
                        ax[i].set_xlim(0., 200.)
                except FileNotFoundError as e:
                    print(e)
                    continue
            if i == 0:
                ax[i].set_ylabel("Proportion of Trials Localized", fontsize=12)
            ax[i].set_xlabel("Distance Travelled (m)", fontsize=12)
            ax[i].set_title(f"{query}", fontsize=16)
    ax[-1].legend()
    # sort legend labels
    handles, labels = ax[-1].get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax[-1].legend(handles, labels)
    # resize figures, larger
    old_fig_size = fig.get_size_inches()
    fig.set_size_inches(old_fig_size[0] * 3.0, old_fig_size[1] * 1.5)
    fig.suptitle("RobotCar: Proportion Localized by Distance (m)", fontsize=20)
    fig.tight_layout()

    plt.savefig(path.join(RESULTS_DIR, f"propn_by_dist_{args.transl_err:.0f}m"
                          f"_{args.rot_err:.0f}deg.pdf"))

    ###### Second figure, median distance to localize by precision ######

    fig1, ax1 = plt.subplots(1, len(queries), sharey=True)

    for i, query in enumerate(queries):
        minx = np.inf
        for method in methods:
            try:
                desc = descs[query+method]
                results = preprocess_results(path.join(RESULTS_DIR, desc,
                                                       'results.csv'))
                # baseline method has no variable convergence threshold,
                # so do not plot curve and output summary statistics
                if method != "Baseline":
                    x, y = prec_by_median_dist(results, args.transl_err,
                                               args.rot_err)
                    # truncate x-axis for plotting
                    if x[-1] < minx:
                        minx = x[-1]
                    # plot curves
                    ax1[i].plot(x, y, label=method, color=colors[method],
                                linewidth=3)
            except FileNotFoundError as e:
                print(e)
                continue
        # truncate x-axis for each traverse
        ax1[i].set_xlim(0., minx)
        if i == 0:
            ax1[i].set_ylabel("Precision", fontsize=12)
        # formatting, titles, labels
        ax1[i].set_xlabel("Median Distance to Convergence (m)", fontsize=12)
        ax1[i].set_title(f"{query}", fontsize=16)
    ax1[-1].legend()
    # sort legend labels
    handles, labels = ax1[-1].get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax1[-1].legend(handles, labels)
    # resize plots (larger)
    old_fig_size1 = fig1.get_size_inches()
    fig1.set_size_inches(old_fig_size1[0] * 3.0, old_fig_size1[1] * 1.5)
    fig1.suptitle("RobotCar: Precision by Median Distance to Converge (m)", fontsize=20)
    fig1.tight_layout()
    plt.savefig(path.join(RESULTS_DIR, f"prec_by_dist_{args.transl_err:.0f}m"
                          f"_{args.rot_err:.0f}deg.pdf"))

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as path

import pandas as pd

from settings import RESULTS_DIR

self_dirpath = os.path.dirname(os.path.abspath(__file__))


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
        print(score, P, prec_lvl)
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
    print("scoeasadas" + str(scoreP))
    _, _, dist_from_start = localize_with_score(results, scoreP,
                                                baseline=baseline)
    x = np.sort(dist_from_start)
    y = 1. - np.linspace(0., 1., len(x))  # empirical cdf
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Aggregate results for comparison tables"))
    parser.add_argument("-f", "--filename", type=str, default='results_alloc.csv',
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

    # generate result curve data
    for prec in args.precision_levels:
        for query in queries:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(f"{query} @ {prec}")
            for method in methods:
                print("method" + query + method + str(prec))
                try:
                    baseline = (method == "Baseline")
                    desc = descs[query+method]
                    results = preprocess_results(path.join(RESULTS_DIR, desc,
                                                      'results.csv'))
                    x, y = loc_curve_at_prec(results, prec, args.transl_err,
                                             args.rot_err, baseline=baseline)
                    ax.plot(x, y, label=method)
                except FileNotFoundError as e:
                    print(e)
                    continue
            ax.legend()
        plt.show()


    # plot curves and save

import argparse
import csv
import numpy as np
import os
import os.path as path

import pandas as pd

from settings import RESULTS_DIR


def preprocess_results(results_path):
    """
    reads raw results from results.csv files and turns processes them
    into a form more usable for evaluation
    Args:
        results_path: path of results.csv file
    returns:
        results: processed results
    """
    results = []
    with open(results_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(reader):
            # skip header
            if i == 0:
                continue
            # recover results from row
            row_results = np.asarray(row[2:], dtype=np.float32).reshape(-1, 2)
            t_errs = row_results[:, 0]
            R_errs = row_results[:, 1]
            # save to list
            results.append((t_errs, R_errs))
    return results


def precision_at_tol(results, t_thres, R_thres):
    """
    Compute precision of results given ground truth error tolerances
    Args:
        results: processed results in a list outputted by the
                 preprocess_results function
        t_thres: threshold on translational error (m) for trial success
        R_thres: threshold on orientation error (deg) for trial success
    Returns:
        precisions: array of precision [0, 1] for all trials
    """
    precisions = []
    for t_errs, R_errs in results:
        success = np.logical_and(t_errs < t_thres, R_errs < R_thres)
        precisions.append(success.sum() / len(success) * 100.)
    return precisions


self_dirpath = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Aggregate results for comparison tables"))
    parser.add_argument("-f", "--filename", type=str, default='results_batch.csv',
                    help="filename containing result descriptions to aggregate")
    parser.add_argument("-t", "--transl-err", type=float, default=5.,
                        help="translational error (m) threshold for success")
    parser.add_argument("-R", "--rot-err", type=float, default=30.,
                        help="rotational error threshold (deg) for success")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print detailed results")
    args = parser.parse_args()

    df_desc = pd.read_csv(path.join(self_dirpath, args.filename))
    df_desc_rows = df_desc.values.tolist()
    # save result filenames for lookup during plotting
    descs = {}
    for query, method, desc in df_desc_rows:
        descs[query+method] = desc
    # extract all methods to be evaluated
    methods = set(row[1] for row in df_desc_rows if row[1] != "Baseline")
    queries = set(row[0] for row in df_desc_rows)

    rows = []

    for i, query in enumerate(queries):
        for method in methods:
            try:
                desc = descs[query+method]
                results = preprocess_results(path.join(RESULTS_DIR, "batch",
                                                       desc, 'results.csv'))
                # baseline method cannot do batch estimation
                precisions = precision_at_tol(results, args.transl_err,
                                              args.rot_err)
                rows.extend([(trial+1, query, method, precision) for
                             trial, precision in enumerate(precisions)])
            except FileNotFoundError as e:
                print(e)
                continue

    df = pd.DataFrame(rows, columns=['Trial', 'Query', 'Method', 'Precision'])
    agg_pivot = pd.pivot_table(df, values="Precision", index=["Method"],
                               columns=["Query"], aggfunc=np.mean).round(1)

    # print detailed results log if verbose

    if args.verbose:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(f"Summary results, Mean precision at "
              f"{args.transl_err:.0f}m {args.rot_err:.0f} deg\n", agg_pivot, "\n")

    # save results to disk

    agg_pivot.to_latex(path.join(RESULTS_DIR, 'batch',
                'avg_precision_' + args.filename[:-4] + '.tex'), index=True)

    # save to disk, each query traverse has separate sheet in same ssheet

    detailed_fpath = path.join(RESULTS_DIR, 'batch', 'detailed_precision_'
                               + args.filename[:-4] + '.xlsx')
    with pd.ExcelWriter(detailed_fpath) as writer:
        for query in queries:
            query_pivot = pd.pivot_table(df[df["Query"] == query],
                                         values="Precision", index=["Trial"],
                                         columns=["Method"], aggfunc=np.sum)
            if args.verbose:
                print(f"Traverse: {query}, Mean precision at "
                      f"{args.transl_err:.0f}m {args.rot_err:.0f} deg\n",
                      query_pivot.round(1), "\n")

            query_pivot.round(1).to_excel(writer, sheet_name=query)

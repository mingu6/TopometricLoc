import argparse
import os
import os.path as path

import pandas as pd

from settings import RESULTS_DIR

self_dirpath = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Aggregate results for comparison tables"))
    parser.add_argument("-f", "--filename", type=str, default='results_alloc.csv',
                    help="filename containing result descriptions to aggregate")
    parser.add_argument("-t", "--transl_err", type=float, default=5.,
                        help="translational error (m) threshold for success")
    parser.add_argument("-R", "--rot_err", type=float, default=30.,
                        help="rotational error threshold (deg) for success")
    args = parser.parse_args()

    df_desc = pd.read_csv(path.join(self_dirpath, args.filename))
    df_desc_rows = df_desc.values.tolist()

    results = []

    for query, method, desc in df_desc_rows:
        df = pd.read_csv(path.join(RESULTS_DIR, desc, 'results.csv'))
        # compute precision
        correct = (df['t_err'] <= args.transl_err) & (df['R_err'] <= args.rot_err)
        precision = correct.to_numpy().sum() / len(correct)
        # compute median distance to localize
        dist_med = df['dist_to_converge'].median()
        results.append((query, method, precision, dist_med))

    # store aggregated results into dataframe

    results_df = pd.DataFrame(
        results, columns=['Query', 'Method', 'Precision', 'Dist. to localize'])
    precision_pivot = (results_df.pivot(columns='Query', index='Method',
                                        values='Precision') * 100).round(1)
    dist_to_loc_pivot = results_df.pivot(columns='Query', index='Method',
                                         values='Dist. to localize').round(1)
    precision_pivot.to_latex(path.join(
        RESULTS_DIR, 'precision_' + args.filename[:-4] + '.tex'), index=True)
    dist_to_loc_pivot.to_latex(path.join(
        RESULTS_DIR, 'dist_to_loc_' + args.filename[:-4] + '.tex'), index=True)

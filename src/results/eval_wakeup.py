import argparse
import os.path as path
import numpy as np

import pickle

from settings import RESULTS_DIR


def main():
    fname = path.join(RESULTS_DIR, 'wakeup', 'overcast1_xy_1_t_10_wd_4_night-rain_xy_2_t_15_noverif_1',
              'results.pickle')
    with open(fname, "rb") as f:
        results = pickle.load(f)
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Aggregate results for comparison tables"))
    parser.add_argument("-f", "--filename", type=str, default='results_online.csv',
                    help="filename containing result descriptions to aggregate")
    parser.add_argument("-t", "--transl-err", type=float, default=5.,
                        help="translational error (m) threshold for success")
    parser.add_argument("-R", "--rot-err", type=float, default=30.,
                        help="rotational error threshold (deg) for success")
    args = parser.parse_args()

    main()

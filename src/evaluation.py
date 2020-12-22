import argparse
import os
import os.path as path
import pickle

import yaml

from data.utils import load_pose_data
from mapping import RefMap
from localization import Localization
from settings import DATA_DIR

self_dirpath = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-d", "--description", type=str, default="default",
                        help="description of model for experiment")
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverse", type=str, default="night",
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str, default='t_1_w_10_wd_3.pickle',
                    help="filename containing subsampled reference traverse poses")
    parser.add_argument("-qf", "--query-filename", type=str, default='t_1_w_10.csv',
                    help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="default.yaml",
                        help="filename containing model parameters")
    args = parser.parse_args()

    query_traverse = args.query_traverse
    q_fname = args.query_filename

    # read in parameters
    params_path = path.abspath(path.join(self_dirpath, "params"))
    with open(path.join(params_path, args.params), 'r') as f:
        params = yaml.load(f)

    # load map
    map_dir = path.join(DATA_DIR, args.reference_traverse, 'saved_maps')
    fpath = path.join(map_dir, args.reference_filename)
    with open(fpath, "rb") as f:
        refMap = pickle.load(f)

    # load query sequence
    tstampsQ, xyzrpyQ, odomQ = load_pose_data(query_traverse, q_fname)

    # setup localization object
    loc = Localization(params, refMap)
    loc._update_motion(odomQ[0])

import argparse
from functools import reduce
import os
import os.path as path
import numpy as np
import pickle
import time
from tqdm import tqdm

import pandas as pd
from scipy.sparse import csc_matrix
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra

from build_reference_map import read_descriptors, build_map, load_subsampled_data
from motion_model import shortest_dist_segments
from settings import DATA_DIR
import geometry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverse", type=str, default="night",
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str, required=True,
                    help="filename containing subsampled reference traverse poses")
    parser.add_argument("-qf", "--query-filename", type=str, required=True,
                    help="filename containing subsampled query traverse poses")
    parser.add_argument("-w", "--attitude-weight", type=float, default=10,
        help=("weight for attitude component of pose distances equal to d where"
              "1 / d being rotation angle (rad) equivalent to 1m translation"))
    parser.add_argument("-s", "--subsample-threshold", type=float, default=3,
        help=("distance travelled before adding a sequence element"))
    parser.add_argument("-p", "--pca-dim", type=int, default=1024,
                        help="number of dimensions for nv descriptor")
    args = parser.parse_args()

    w = args.attitude_weight;
    ref_traverse = args.reference_traverse
    query_traverse = args.query_traverse
    ref_fname = args.reference_filename
    q_fname = args.query_filename
    pca_dim = args.pca_dim

    att_wt = np.ones(6)
    att_wt[3:] *= w

    # load/build reference map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, f'{ref_fname[:-4]}_wd_{w:.0f}.pickle')

    try:
        with open(fpath, "rb") as f:
            ref_map = pickle.load(f)
    except FileNotFoundError as err:
        print("cached map file not found, building map...")
        tstamps, poses, descriptors = load_subsampled_data(ref_traverse, ref_fname, pca_dim)
        ref_map = build_map(ref_traverse, tstamps, poses, descriptors,
                            w, 2 * args.subsample_threshold)
        with open(fpath, "rb") as f:
            pickle.dump(ref_map, f)
        print(f"successfully saved map as {fpath}")
    # add off map node
    ref_map.add_node("off")
    N = len(ref_map)

    # load query sequence
    tstampsQ, posesQ, descriptorsQ = load_subsampled_data(query_traverse, q_fname, pca_dim)
    odomQ = (posesQ[:-1] / posesQ[1:]).to_xyzrpy()
    odomQ *= att_wt[None, :]

    # initialize belief
    p_off_prior = 0.1
    belief = np.ones(N) * (1 - p_off_prior) / N
    belief[-1] = p_off_prior

    # initialize components for motion model calculations

    source, dest, tO1, tO2, tD1, tD2, tOD = \
        zip(*[(s, d, data["tO1"], data["tO2"],
               data["tD1"], data["tD2"], data["tOD"])
              for (s, d, k, data) in ref_map.edges.data(keys=True)
              if k == "nonself"])

    tO1 = np.asarray(tO1) * att_wt
    tO2 = np.asarray(tO2) * att_wt
    tD1 = np.asarray(tD1) * att_wt
    tD2 = np.asarray(tD2) * att_wt
    tOD = np.asarray(tOD) * att_wt

    # loop through sequence and update belief
    start_ind = 200
    end_ind = 220
    for i in range(start_ind, end_ind):
        # compute consistency score
        d11 = shortest_dist_segments(tO1 + odomQ[i], -tO1,
                                     tD1, tOD - tD1)
        d12 = shortest_dist_segments(tO1 + odomQ[i], -tO1,
                                     tOD, tD2 - tOD)
        d21 = shortest_dist_segments(odomQ[i], tO2,
                                     tD1, tOD - tD1)
        d22 = shortest_dist_segments(odomQ[i], tO2,
                                     tOD, tD2 - tOD)
        d = np.stack((d11, d12, d21, d22), axis=1).min(axis=1)
        E = csc_matrix((np.exp(-d), (source, dest)), shape=(N, N))
        dmin = np.squeeze(np.asarray(E.max(axis=1).toarray()))

        import matplotlib.pyplot as plt
        tstamps, poses, descriptors = load_subsampled_data(ref_traverse, ref_fname, pca_dim)
        xyzrpy = poses.to_xyzrpy()
        print(odomQ[i])
        plt.scatter(xyzrpy[:, 1], xyzrpy[:, 0], c=dmin[:-1])
        plt.colorbar()
        plt.show()
        import pdb; pdb.set_trace()
        #drel_mat = nx.convert_matrix.to_scipy_sparse_matrix(ref_map, weight='d', format='csc')

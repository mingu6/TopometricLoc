import argparse
from functools import reduce
import os
import os.path as path
import numpy as np
import pickle
import time
from tqdm import tqdm

import pandas as pd
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra


from settings import DATA_DIR
import geometry


def load_subsampled_data(traverse, fname, pca_dim, ind=None):
    df = pd.read_csv(path.join(DATA_DIR, traverse, 'subsampled', fname))
    ind_end = len(df) if ind is None else ind
    tstamps = df['timestamp'][:ind_end].to_numpy()
    xyzrpy = df[['northing', 'easting', 'down', 'roll', 'pitch', 'yaw']].to_numpy()
    poses = geometry.SE3.from_xyzrpy(xyzrpy[:ind_end])
    descriptors = read_descriptors(traverse, tstamps)[:ind_end, :pca_dim]
    return tstamps, poses, descriptors


def read_descriptors(traverse, tstamps):
    dirpath = path.join(DATA_DIR, traverse, 'features/global')
    descriptors = [np.load(path.join(dirpath, f'{ts}.npy')) for ts in tstamps]
    return np.concatenate(descriptors, axis=0)


def build_map(traverse, tstamps, poses, descriptors, w, max_dist):
    mapG = nx.MultiDiGraph()
    N = len(tstamps)

    # add nodes

    node_inds = np.arange(N)
    nodes = [(ind, {"nv": descriptors[ind]}) for ind in node_inds]
    mapG.add_nodes_from(nodes)

    # add neighbor edges b/w subsequent frames (loop closures in future)

    att_wt = np.ones(6)
    att_wt[3:] *= w

    for ind in tqdm(node_inds, "nhood edges"):

        # identify nearby nodes and extract relative poses

        nhood = np.arange(ind, min(ind+2, N))
        rel_xyzrpy = np.atleast_2d((poses[ind] / poses[nhood]).to_xyzrpy())
        drel = np.linalg.norm(rel_xyzrpy * att_wt, axis=1)

        # insert self-edges and immediate neighbor edges (nhood map)

        edges = [(ind, i, "nh", {"rpose": rel_xyzrpy[i-ind], "d": drel[i-ind]})
                 for i in nhood]

        mapG.add_edges_from(edges)

    # create transition edges (window)

    #allpairs = all_pairs_dijkstra(mapG, cutoff=max_dist, weight='d')
    allpairs = all_pairs_dijkstra(mapG, cutoff=3)

    for source, (_, paths) in tqdm(allpairs, desc="transition edges", total=N):
        for dest in paths.keys():
            if dest == source:
                # self-transition case: assume node represents a line segment
                # between two midpoints: one between node and predecessor and
                # one between node and successor node.
                if source == 0:
                    start = - mapG[0][1]["nh"]["rpose"] * 0.5
                    end = mapG[0][1]["nh"]["rpose"] * 0.5
                elif source == N-1:
                    start = -0.5 * mapG[N-2][N-1]["nh"]["rpose"]
                    end = 0.5 * mapG[N-2][N-1]["nh"]["rpose"]
                else:
                    start = 0.5 * (poses[source] / poses[source-1]).to_xyzrpy()
                    end = 0.5 * (poses[source] / poses[source+1]).to_xyzrpy()
                mapG.add_edge(source, dest, "self", tO1=start, tO2=end)
            else:
                # regular transition case: draw line segment around origin node
                # and another around the destination node. odometry is then
                # compared to lines between all points on these line segments
                tO1 = 0.5 * (poses[source] / poses[source-1]).to_xyzrpy() if \
                        source > 0 else -0.5 * mapG[0][1]["nh"]["rpose"]
                tO2 = 0.5 * (poses[source] / poses[source+1]).to_xyzrpy()

                tD1 = 0.5 * ((poses[source] / poses[dest]).to_xyzrpy() +
                             (poses[source] / poses[dest-1]).to_xyzrpy())
                if dest < N-1:
                    tD2 = 0.5 * ((poses[source] / poses[dest]).to_xyzrpy() +
                                 (poses[source] / poses[dest+1]).to_xyzrpy())
                else:
                    # TO DO: VERFIY THIS TRANSITION LARGE
                    tD2 = 1.5 * (poses[source] / poses[dest]).to_xyzrpy() - \
                            0.5 * (poses[source] / poses[dest-1]).to_xyzrpy()
                tOD = (poses[source] / poses[dest]).to_xyzrpy()

                mapG.add_edge(source, dest, "nonself",
                              tO1=tO1, tO2=tO2, tD1=tD1, tD2=tD2,
                              tOD=tOD)

    return mapG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-t", "--traverse", type=str, default="overcast",
                        help="traverse name, e.g. overcast, night")
    parser.add_argument("-f", "--filename", type=str, required=True,
                    help="filename containing subsampled traverse poses")
    parser.add_argument("-w", "--attitude-weight", type=float, default=10,
        help=("weight for attitude component of pose distances equal to d where"
              "1 / d being rotation angle (rad) equivalent to 1m translation"))
    parser.add_argument("-d", "--max-dist", type=float, default=7,
        help=("maximum distance for possible transition between nodes"))
    parser.add_argument("-p", "--pca-dim", type=int, default=1024,
                        help="number of dimensions for nv descriptor")
    args = parser.parse_args()

    w = args.attitude_weight
    d = args.max_dist
    traverse = args.traverse
    fname = args.filename

    # read reference map node data
    tstamps, poses, descriptors = load_subsampled_data(traverse, fname, args.pca_dim)

    ref_map = build_map(traverse, tstamps, poses, descriptors, w, d)

    # save map
    map_dir = path.join(DATA_DIR, traverse, 'saved_maps')
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    fname = f"{fname[:-4]}_wd_{w:.0f}"

    with open(path.join(map_dir, f"{fname}.pickle"), "wb") as f:
        pickle.dump(ref_map, f)

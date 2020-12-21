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
    vo = df[['vo_x', 'vo_y', 'vo_z', 'vo_roll', 'vo_pitch', 'vo_yaw']].to_numpy()
    poses = geometry.SE3.from_xyzrpy(xyzrpy[:ind_end])
    loc_des, glb_des = read_descriptors(traverse, tstamps)
    return tstamps, poses, vo, loc_des[:ind_end], glb_des[:ind_end, :pca_dim]


def read_descriptors(traverse, tstamps):
    g_dirpath = path.join(DATA_DIR, traverse, 'features/global')
    l_dirpath = path.join(DATA_DIR, traverse, 'features/local')

    glb_des = []
    local_des = []

    for ts in tqdm(tstamps, desc=f"loading descriptors {traverse}"):
        # global descriptors
        glb_des.append(np.load(path.join(g_dirpath, f'{ts}.npy')))
        # local descriptors
        with np.load(path.join(l_dirpath, f'{ts}.npz')) as f:
            local_des.append(dict(f))
    glb_des = np.concatenate(glb_des, axis=0)
    return local_des, glb_des


def build_map(traverse, tstamps, poses, vo, descriptors, width):
    mapG = nx.MultiDiGraph()
    N = len(tstamps)

    # preprocess odom
    vo_se3 = geometry.SE3.from_xyzrpy(vo)
    cum_odom = [geometry.SE3.from_xyzrpy(np.zeros(6))]
    for i in range(1, len(vo_se3)+1):
        cum_odom.append(cum_odom[-1] * vo_se3[i-1])
    cum_odom = geometry.combine(cum_odom)

    # add nodes

    node_inds = np.arange(N)
    nodes = [(ind, {"nv": descriptors[ind], "gt": poses[ind].to_xyzrpy()})
             for ind in node_inds]
    mapG.add_nodes_from(nodes)

    # add neighbor edges b/w subsequent frames (loop closures in future)

    for source in tqdm(node_inds, "edges"):

        # identify nearby nodes and extract relative poses

        nhood = np.arange(source, min(source+width, N))

        # insert self-edges and immediate neighbor edges (nhood map)

        for dest in nhood:
            if dest == source:
                # self-transition case: assume node represents a line segment
                # between two midpoints: one between node and predecessor and
                # one between node and successor node.
                if source == 0:
                    start = - vo[0] * 0.25
                    end = vo[0] * 0.25
                elif source == N-1:
                    start = -0.25 * vo[N-1]
                    end = 0.25 * vo[N-1]
                else:
                    start = 0.25 * geometry.SE3.from_xyzrpy(
                        vo[source-1]).inv().to_xyzrpy()
                    end = 0.25 * vo[source]
                mapG.add_edge(source, dest, "tr", s1=start, s2=end)
            else:
                # regular transition case: node is origin, transition to
                # line segment around source node where the segment is
                # given by average odometry

                relpose = (cum_odom[source] / cum_odom[dest]).to_xyzrpy()
                relpose_bw = (cum_odom[source] / cum_odom[dest-1]).to_xyzrpy()
                s1 = 0.5 * (relpose + relpose_bw)
                if dest < N-1:
                    relpose_fw = (cum_odom[source] / cum_odom[dest+1]).to_xyzrpy()
                    s2 = 0.5 * (relpose + relpose_fw)
                else:
                    s2 = 1.5 * relpose - 0.5 * relpose_bw
                mapG.add_edge(source, dest, "tr", s1=s1, s2=s2)

    return mapG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-t", "--traverse", type=str, default="overcast1",
                        help="traverse name, e.g. overcast, night")
    parser.add_argument("-f", "--filename", type=str, required=True,
                    help="filename containing subsampled traverse poses")
    parser.add_argument("-w", "--width", type=int, default=8,
        help=("maximum distance for possible transition between nodes"))
    parser.add_argument("-p", "--pca-dim", type=int, default=4096,
                        help="number of dimensions for nv descriptor")
    args = parser.parse_args()

    traverse = args.traverse
    fname = args.filename

    # read reference map node data
    tstamps, poses, vo, descriptors = load_subsampled_data(traverse, fname, args.pca_dim)

    ref_map = build_map(traverse, tstamps, poses, vo, descriptors, args.width)

    # save map
    map_dir = path.join(DATA_DIR, traverse, 'saved_maps')
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    fname = f"{fname[:-4]}_wd_{args.width}"

    with open(path.join(map_dir, f"{fname}.pickle"), "wb") as f:
        pickle.dump(ref_map, f)

import argparse
import os
import os.path as path
import numpy as np
import pickle

import cv2
import pandas as pd

from data.utils import read_global, load_pose_data, read_local
from settings import DATA_DIR


def odom_segments(odom, width):
    """
    For each state transition in map, generate start/end line segment
    representing set of possible relative poses for transition. In paper
    notation, u_{i\rightarrow j^*_-} and u_{i\rightarrow j^*_+}.

    Output should be an N x width x 2 x 3 matrix, where N is the number
    of nodes, width is the number of transitions from each node.

    Args:
        odom: Relative odometry between reference images, (N-1 x 3) np array
        width: Number of outgoing connections (state transitions) from each
        node
    Output:
        segments: N x width x 2 x 3 np array with relative pose segments as
        described above.
    """
    N = len(odom) + 1  # N nodes has N-1 relative poses between
    # entry i, d of relpose contains relative pose from odom i, i+d for d <= width
    # note: last node has no outward connections
    relpose = np.zeros((N, width+2, 3))
    # for each source node/connection compute relative pose
    # include one node more than width for endpoint for furthest transition
    for w in range(width+2):
        if w == 1:
            # relative pose given by raw odom (1 node away transition)
            relpose[:-1, w] = odom
        elif w > 1:
            relpose[:-w, w] = relpose[:-w, w-1] + odom[w-1:]

    # compute start/end segments for non-self transitions
    segments = np.zeros((N, width+1, 2, 3))
    # start segment j^*_-: 
    segments[:, 1:, 0, :] = 0.5 * (relpose[:, :-2, :] + relpose[:, 1:-1, :])
    # end segment j^*_+
    segments[:, 1:, 1, :] = 0.5 * (relpose[:, 1:-1, :] + relpose[:, 2:, :])
    # adjust end-segments for nodes near final node (no successors, use neg predecessor)
    for ni in range(N-width, N):
        gap = N - ni - 1
        segments[ni, gap+1, 0, :] = 0.  # relative pose computed beyond end node above
        segments[ni, gap, 1, :] = 1.5 * relpose[ni, gap, :] - 0.5 * relpose[ni, gap-1, :]
    # adjust self-transitions (smaller region around source node), successor segment
    segments[1:, 0, 0, :] = -0.25 * odom  # transition from predecessor (basis change)
    segments[0, 0, 0, :] = -0.25 * odom[0]  # flip forward transition from root node
    # predecessor segment
    segments[:-1, 0, 1, :] = 0.25 * odom  # forward transition just odom
    segments[-1, 0, 1, :] = 0.25 * odom[-1]  # no successor for last node, use predecessor
    return segments


class RefMap:
    def __init__(self, traverse, tstamps, odom, width=3, gt_poses=None):
        # parameters/metadata
        self.N = len(tstamps)
        self.traverse = traverse
        self.width = width  # number of connections from each node
        # store traverse node information
        self.tstamps = tstamps
        self.glb_des = read_global(traverse, tstamps)
        self.odom_segments = odom_segments(odom, width)
        # ground truth poses
        self.gt_poses = gt_poses

    def load_local(self, ind, num_feats=None):
        """
        loads local descriptors (keypoints, descriptors) from disk
        """
        kp, des = read_local(self.traverse, self.tstamps[ind],
                             num_feats=num_feats)
        return kp, des

    def load_image(self, ind):
        """
        loads relevant reference image from disk
        """
        img = cv2.imread(path.join(DATA_DIR, self.traverse,
                                   'images/left', str(self.tstamps[ind]) + '.png'))
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-t", "--traverse", type=str, default="overcast1",
                        help="traverse name, e.g. overcast, night")
    parser.add_argument("-f", "--filename", type=str, required=True,
                    help="filename containing subsampled traverse poses")
    parser.add_argument("-w", "--width", type=int, default=3,
        help=("maximum distance for possible transition between nodes"))
    args = parser.parse_args()

    traverse = args.traverse
    fname = args.filename

    # read reference map node data
    tstamps, xyzrpy, vo = load_pose_data(traverse, fname)
    refMap = RefMap(traverse, tstamps, vo, width=args.width, gt_poses=xyzrpy)

    # save map
    map_dir = path.join(DATA_DIR, traverse, 'saved_maps')
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    fname = f"{fname[:-4]}_wd_{args.width}"

    with open(path.join(map_dir, f"{fname}.pickle"), "wb") as f:
        pickle.dump(refMap, f)

import argparse
import os
import os.path as path
import numpy as np
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import yaml

from data.utils import preprocess_robotcar
from data.prob_motion import motion_update
from settings import DATA_DIR, xyz_centre
from geometry import SE3, SE2

self_dirpath = os.path.dirname(os.path.abspath(__file__))


def generate_traverse_robotcar(traverse, params, gt_ts, gt_poses, vo_ts, vo,
                               xy_thres, theta_thres, start_ind=0):
    """
    Generates nodes in reference/query traverses by subsampling raw data at
    regular spatial frequencies based on raw odometry information. For each
    relative pose between nodes, save mean and covariance information from
    linearized probabilistic motion model.

    Args:
        traverse: traverse name (str)
        params: motion model parameters (alpha1, ..., alpha4)
        gt_ts: ground truth (i.e. RTK or INS) pose
        gt_poses: ground truth RTK poses in SE3 object format
        vo_ts: visual odometry timestamps (source, dest)
        vo: relative poses between source, dest timestamps
        xy_thres: threshold on translation before creating a new node
        theta_thres: threshold on orientation before creating a new node
        start_ind (optional): start index for generating trajectory
    Returns:

    NOTE:

    Each vo and gt are aligned. vo[i] is the relative pose from odometry
    between gt[i-1] and gt[i]

    Each node is the origin of the local coordinate frame. Pose estimate
    at creation of node is point est. at origin. Incorporate VO
    observations incrementally, updating state est. and uncertainty
    relative to previous node until robot has travelled far enough and
    create a new node. Save relative pose estimate and uncertainty between
    previous and new nodes as probabilistic odomety.

    VO can be very buggy for some traverses, large glitches where there
    is a large motion estimate over a very small time interval. Usually,
    these manifest as large values in the y coordinate in Euler form.
    This is addressed manually in this function.

    Also, the dusk RTK malfunctions after a particular timestamp partway
    through the traverse, this is adjusted for manually i.e. stop adding
    more frames past the timestamp.
    """
    ts_start = vo_ts[start_ind]  # start ind based on VO
    poses_start = np.squeeze(np.where(gt_ts == ts_start))

    # setup first node
    tstamps = [ts_start]
    gt_traverse = [gt_poses[poses_start]]  # subsampled adjusted poses
    vo_traverse = [(np.zeros(3), np.zeros((3, 3)))]  # no predecessor
    # initialize relative pose from first node
    mu_accum = np.zeros(3)
    Sigma_accum = np.zeros((3, 3))

    for i in tqdm(range(start_ind, len(vo_ts)), leave=False):
        # if moved far enough, create new node
        xy_mag = np.linalg.norm(mu_accum[:2])
        theta_mag = np.abs(mu_accum[2]) * 180 / np.pi
        if xy_mag > xy_thres or theta_mag > theta_thres:
            curr_pose = np.squeeze(gt_poses[np.where(gt_ts == vo_ts[i])])
            # if image is missing, but vo entry exists, skip to next frame
            if len(curr_pose) == 0:
                continue

            tstamps.append(vo_ts[i])
            gt_traverse.append(curr_pose)  # save gt
            vo_traverse.append((mu_accum, Sigma_accum))

            # new node, reset odometry to point pose at origin
            mu_accum = np.zeros(3)
            Sigma_accum = np.zeros((3, 3))
        # update state estimate
        mu_accum, Sigma_accum = motion_update(params, vo[i],
                                              mu_accum, Sigma_accum)

    tstamps = np.asarray(tstamps, dtype=np.int64)
    gt_traverse = np.vstack(gt_traverse)

    # remove first two observations since camera + rtk wigs out
    # on initialization, yielding garbage vo and/or rtk readings
    vo_traverse[2][0][:] = 0.
    vo_traverse[2][1][...] = 0.
    return tstamps[2:], gt_traverse[2:], vo_traverse[2:]


def save_traverse(traverse, xy_thres, theta_thres,
                  ts_trav, gt_trav, odom_trav):
    traverse_path = path.join(DATA_DIR, traverse)

    # save timestamps of subsampled traverse to disk

    save_path = path.join(traverse_path, "subsampled")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    state = ['x', 'y', 'theta']
    columns = ['ts'] + state + ['mu_x', 'mu_y', 'mu_theta'] + \
        [f"Sigma_{i}{j}" for i in state for j in state]
    odom_mu, odom_Sigma = zip(*odom_trav)
    data = np.hstack((ts_trav[:, None], gt_trav, np.asarray(odom_mu),
                      np.asarray(odom_Sigma).reshape(-1, 9)))
    df = pd.DataFrame(data, columns=columns)
    df = df.astype({"ts": int})
    fout = path.join(save_path,
                     f'xy_{xy_thres:.0f}_t_{theta_thres:.0f}.csv')
    df.to_csv(fout, index=False)
    return None


def load_params_robotcar(params_fname, traverse):

    if params_fname:
        params_file = params_fname
    else:
        params_file = "ours.yaml"

    params_path = path.abspath(path.join(self_dirpath, "..", "params"))
    with open(path.join(params_path, params_file), 'r') as f:
        params = yaml.safe_load(f)

    if traverse == 'night':
        motion_params = np.asarray(params['motion']['night'])
    else:
        motion_params = np.asarray(params['motion']['other'])

    return motion_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Spatially subsample a traverse for map building."))
    parser.add_argument("-t", "--traverses", nargs="+", type=str,
        default=["overcast1"], help=(
            "Names of reference traverses to process,"
            "e.g. overcast, rain, dusk  etc. Input 'all'"
            "instead to process all available references traverses."))
    parser.add_argument("-xy", "--xy-thres", type=float, default=1,
        help="threshold on mean translation for node creation")
    parser.add_argument("-tt", "--theta-thres", type=float, default=10,
        help="threshold on mean orientation (degrees) for node creation")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    args = parser.parse_args()

    if "all" in args.traverses:
        traverses = [traverse for traverse in os.listdir(DATA_DIR)]
    else:
        traverses = args.traverses

    for traverse in tqdm(traverses, leave=False):

        # load gt, odometry and tstamp data

        gt_tstamps, gt_poses, odom_ts, odom = preprocess_robotcar(traverse)

        # if RobotCar, load saved motion model parameters

        params = load_params_robotcar(args.params, traverse)

        # subsample traverse using increments based on ground truth poses

        ts_trav, gt_trav, odom_trav = generate_traverse_robotcar(
            traverse, params, gt_tstamps, gt_poses, odom_ts, odom,
            args.xy_thres, args.theta_thres)

        # save traverse data to disk
        save_traverse(traverse, args.xy_thres, args.theta_thres,
                      ts_trav, gt_trav, odom_trav)

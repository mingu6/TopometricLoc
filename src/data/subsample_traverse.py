import argparse
import os
import os.path as path
import numpy as np
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

from settings import DATA_DIR, xyz_centre
import geometry


def spatial_subsample(timestamps, poses, vo_ts, vo,
                      subsample_threshold, attitude_weight,
                      start_ind=0):
    T1 = geometry.SE3.from_xyzrpy(np.asarray([0, 0, 0, -np.pi, -np.pi, np.pi / 2]))
    poses = poses * T1  # rotates coord frame so fw motion is in the x coord

    ts_start = vo_ts[start_ind, 1]  # start ind based on VO
    poses_start = np.where(timestamps == ts_start)

    tstamps = [ts_start]
    xyzrpy = [poses[poses_start].to_xyzrpy()]  # subsampled adjusted poses
    xyzrpy[0][:3] -= xyz_centre  # adjust coordinate frame for smaller values
    vo_sub = []
    vo_accum = geometry.SE3.from_xyzrpy(np.zeros(6))

    pose_temp = poses[start_ind]  # most recent subsampled node
    for i in tqdm(range(start_ind, len(vo_ts)), leave=False):
        t_mag, R_mag = vo_accum.magnitude()
        curr_diff = t_mag + attitude_weight * R_mag
        if curr_diff > subsample_threshold:
            pose_xyzrpy = poses[np.where(timestamps == vo_ts[i, 1])].to_xyzrpy()
            if not len(pose_xyzrpy):  # image is missing, but vo entry exists, skip to next frame
                continue
            pose_xyzrpy[:3] -= xyz_centre

            tstamps.append(vo_ts[i, 1])
            xyzrpy.append(pose_xyzrpy)  # save gt
            vo_sub.append(vo_accum.to_xyzrpy())

            vo_accum = geometry.SE3.from_xyzrpy(np.zeros(6))
        vo_accum *= vo[i]  # accumulate odometry

    vo_sub.append(np.zeros(6))
    tstamps = np.asarray(tstamps, dtype=np.int64)
    xyzrpy = np.asarray(xyzrpy)
    vo_sub = np.asarray(vo_sub)

    return tstamps, xyzrpy, vo_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Spatially subsample a traverse for map building."))
    parser.add_argument("-t", "--traverses", nargs="+", type=str,
        default=["overcast"], help=(
            "Names of reference traverses to process,"
            "e.g. overcast, rain, dusk  etc. Input 'all'"
            "instead to process all available references traverses."))
    parser.add_argument("-w", "--attitude-weight", type=float, default=5,
        help=("weight for attitude component of pose"
              "distance equal to d where 1 / d being rotation"
              "angle (rad) equivalent to 1m translation"))
    parser.add_argument(
        "-s", "--subsample-threshold", type=float, default=2,
        help="threshold on weighted pose distance to subsample at")
    args = parser.parse_args()

    w = args.attitude_weight
    thres = args.subsample_threshold

    columns = ['timestamp', 'northing', 'easting', 'down', 'roll', 'pitch', 'yaw']
    columns_vo = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    if "all" in args.traverses:
        traverses = [traverse for traverse in os.listdir(DATA_DIR)]
    else:
        traverses = args.traverses

    for traverse in tqdm(traverses, leave=False):
        traverse_path = path.join(DATA_DIR, traverse)

        # load GT RTK timestamps and poses from csv

        raw_df = pd.read_csv(path.join(traverse_path, 'camera_poses.csv'))
        tstamps = raw_df['timestamp'].to_numpy()
        xyzrpy = raw_df[columns[1:]].to_numpy()
        poses = geometry.SE3.from_xyzrpy(xyzrpy)

        # load VO timestamps and poses from csv

        vo_df = pd.read_csv(path.join(traverse_path, 'vo.csv'))
        vo_ts = vo_df[['source_timestamp', 'destination_timestamp']].to_numpy()
        vo_xyzrpy = vo_df[columns_vo].to_numpy()
        vo = geometry.SE3.from_xyzrpy(vo_xyzrpy)

        # subsample traverse using increments based on ground truth poses

        tstamps_sub, xyzrpy_sub, vo_sub = spatial_subsample(
            tstamps, poses, vo_ts, vo, thres, w)

        acc_pose = geometry.SE3.from_xyzrpy(xyzrpy_sub[0])
        poses_vo = [acc_pose.to_xyzrpy()]
        for v in vo:
            acc_pose *= v
            poses_vo.append(acc_pose.to_xyzrpy())
        poses_vo = np.asarray(poses_vo)
        #plt.scatter(xyzrpy_sub[:, 1], xyzrpy_sub[:, 0], color='green')
        #plt.scatter(poses_vo[:, 1], poses_vo[:, 0], color='red')
        #plt.show()
        # save timestamps of subsampled traverse to disk

        save_path = path.join(traverse_path, "subsampled")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        new_col_vo = [f"vo_{d}" for d in columns_vo]
        df = pd.DataFrame(np.concatenate(
            (tstamps_sub[:, None], xyzrpy_sub, vo_sub), axis=1),
                          columns=columns + new_col_vo)
        df = df.astype({"timestamp": int})
        fout = path.join(save_path, f't_{thres:.0f}_w_{w:.0f}.csv')
        df.to_csv(fout, index=False)

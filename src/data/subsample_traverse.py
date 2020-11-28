import argparse
import os
import os.path as path
import numpy as np
from tqdm import tqdm

import pandas as pd

from settings import DATA_DIR, xyz_centre
import geometry


def spatial_subsample(timestamps, poses,
                      subsample_threshold, attitude_weight,
                      start_ind=0):
    T1 = geometry.SE3.from_xyzrpy(np.asarray([0, 0, 0, -np.pi, -np.pi, np.pi / 2]))
    poses = poses * T1  # rotates coord frame so fw motion is in the x coord

    tstamps = [timestamps[start_ind]]
    xyzrpy = [poses[start_ind].to_xyzrpy()]  # subsampled adjusted poses
    xyzrpy[0][:3] -= xyz_centre  # adjust coordinate frame for smaller values

    pose_temp = poses[start_ind]  # most recent subsampled node
    for i in tqdm(range(start_ind+1, len(timestamps)-1), leave=False):
        curr_diff = geometry.metric(pose_temp, poses[i], attitude_weight)
        if curr_diff > subsample_threshold:
            pose_temp = poses[i]  # update most recent subsampled node
            pose_xyzrpy = pose_temp.to_xyzrpy()
            pose_xyzrpy[:3] -= xyz_centre

            tstamps.append(timestamps[i])
            xyzrpy.append(pose_xyzrpy)
    tstamps = np.asarray(tstamps, dtype=np.int64)
    xyzrpy = np.asarray(xyzrpy)
    return tstamps, xyzrpy


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

    if "all" in args.traverses:
        traverses = [traverse for traverse in os.listdir(DATA_DIR)]
    else:
        traverses = args.traverses

    for traverse in tqdm(traverses, leave=False):
        traverse_path = path.join(DATA_DIR, traverse)

        # load timestamps and poses from csv

        raw_df = pd.read_csv(path.join(traverse_path, 'camera_poses.csv'))
        tstamps = raw_df['timestamp'].to_numpy()
        xyzrpy = raw_df[columns[1:]].to_numpy()
        poses = geometry.SE3.from_xyzrpy(xyzrpy)

        # subsample traverse using increments based on ground truth poses

        tstamps_sub, xyzrpy_sub = spatial_subsample(tstamps, poses, thres, w)

        # save timestamps of subsampled traverse to disk

        save_path = path.join(traverse_path, "subsampled")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = pd.DataFrame(np.concatenate((tstamps_sub[:, None], xyzrpy_sub), axis=1),
                          columns=columns)
        df = df.astype({"timestamp": int})
        fout = path.join(save_path, f't_{thres:.0f}_w_{w:.0f}.csv')
        df.to_csv(fout, index=False)

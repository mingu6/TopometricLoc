import os.path as path
import numpy as np
import pandas as pd

from settings import DATA_DIR


def load_pose_data(traverse, fname, ind=None):
    df = pd.read_csv(path.join(DATA_DIR, traverse, 'subsampled', fname))
    ind_end = len(df) if ind is None else ind
    tstamps = df['timestamp'][:ind_end].to_numpy()
    #xyzrpy = df[['northing', 'easting', 'down', 'roll', 'pitch', 'yaw']].to_numpy()
    xyzrpy = df[['northing', 'easting', 'yaw']].to_numpy()[:ind_end].astype(np.float32)
    #vo = df[['vo_x', 'vo_y', 'vo_z', 'vo_roll', 'vo_pitch', 'vo_yaw']].to_numpy()[:-1]
    vo = df[['vo_x', 'vo_y', 'vo_yaw']].to_numpy()[:ind_end-1].astype(np.float32)
    return tstamps, xyzrpy, vo


def read_global(traverse, tstamps):
    dirpath = path.join(DATA_DIR, traverse, 'features/global')
    glb_des = np.concatenate([np.load(path.join(dirpath, f'{ts}.npy'))
                              for ts in tstamps], axis=0)
    return glb_des


def preprocess_local_features(features):
    des = features['local_descriptors'].astype(np.float32)[0]
    kp = features['keypoints'].astype(np.int32)
    # remove features detected on the car bonnet
    above_bonnet = kp[:, 1] < 800
    return kp[above_bonnet, :], des[above_bonnet, :]



import os.path as path
import numpy as np

import pandas as pd
import cv2

from ..settings import DATA_DIR, xyz_centre
from ..geometry import SE3, SE2


def load_pose_data(traverse, fname, ind=None):
    df = pd.read_csv(path.join(DATA_DIR, traverse, 'subsampled', fname))
    ind_end = len(df) if ind is None else ind
    tstamps = df['ts'][:ind_end].to_numpy()
    gt_pose = df[['x', 'y', 'theta']].to_numpy()\
        [:ind_end].astype(np.float32)
    vo_mu = df[['mu_x', 'mu_y', 'mu_theta']].to_numpy()\
        [:ind_end].astype(np.float32)
    vo_Sigma = df.iloc[:, -9:].to_numpy().reshape((-1, 3, 3))
    return tstamps, gt_pose, vo_mu, vo_Sigma


def preprocess_robotcar(traverse, cutoffs=None):
    columns = ['timestamp', 'northing', 'easting', 'down',
               'roll', 'pitch', 'yaw']
    columns_vo = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    traverse_path = path.join(DATA_DIR, traverse)

    # load GT RTK timestamps and poses from csv
    # process gt poses by projecting to SE(2) and adjusting centre of
    # translation since RTK gt has large values for translation
    raw_df = pd.read_csv(path.join(traverse_path, 'camera_poses.csv'))
    gt_ts = raw_df['timestamp'].to_numpy()
    gt_poses_SE3 = SE3.from_xyzrpy(raw_df[columns[1:]].to_numpy())
    T1 = SE3.from_xyzrpy(np.asarray([0, 0, 0, -np.pi, -np.pi, np.pi / 2]))
    # rotates coord frame so fw motion is in the x coord
    gt_poses = (gt_poses_SE3 * T1).to_xyzypr()[:, [0, 1, 3]]
    gt_adj = np.array([*xyz_centre[:2], 0.])  # adjust centre of coord frame
    gt_poses = gt_poses - gt_adj[None, :]

    # load VO timestamps and poses from csv

    vo_df = pd.read_csv(path.join(traverse_path, 'vo.csv'))
    vo_ts = vo_df['destination_timestamp'].to_numpy()
    vo_xyzrpy = vo_df[columns_vo].to_numpy()
    vo = SE3.from_xyzrpy(vo_xyzrpy).to_xyzypr()[:, [0, 1, 3]]

    # remove outliers in VO

    outlier_inds = np.abs(vo[:, 1]) > 0.2
    if len(outlier_inds):
        vo = np.delete(vo, outlier_inds, axis=0)
        vo_ts = np.delete(vo_ts, outlier_inds, axis=0)

    # manually cutoff dusk, RTK fails before end of traverse

    dusk_delete_inds_vo = np.logical_and(traverse == 'dusk',
                                         vo_ts > 1416587217921391)
    dusk_delete_inds_gt = np.logical_and(traverse == 'dusk',
                                         gt_ts > 1416587217921391)
    if len(dusk_delete_inds_gt):
        gt_ts = np.delete(gt_ts, dusk_delete_inds_gt, axis=0)
        gt_poses = np.delete(gt_poses, dusk_delete_inds_gt, axis=0)
    if len(dusk_delete_inds_vo):
        vo_ts = np.delete(vo_ts, dusk_delete_inds_vo, axis=0)
        vo = np.delete(vo, dusk_delete_inds_vo, axis=0)

    # accumulate vo until enough distance has been travelled for stable
    # odometry models. Odometry model does not behave well with small/backward
    # motion

    od_accum = SE2(np.zeros(3))
    gt_ts1 = [vo_ts[0]]
    gt_poses1 = [gt_poses[0]]
    vo1 = [np.zeros(3)]
    vo_ts1 = [vo_ts[0]]

    for ts, od in zip(vo_ts, vo):
        xy_mag, th_mag = od_accum.magnitude()
        if xy_mag > 0.5 or th_mag > 5. * np.pi / 180.:
            if ts in gt_ts:
                gt_ts1.append(ts)
                gt_poses1.append(np.squeeze(gt_poses[gt_ts == ts]))
                vo1.append(od_accum.to_vec())
                vo_ts1.append(ts)
                # reset, begin accumulating next obs
                od_accum = SE2(np.zeros(3))
        od_accum = od_accum * SE2(od)

    gt_ts1 = np.asarray(gt_ts1)
    gt_poses1 = np.asarray(gt_poses1)
    vo1 = np.asarray(vo1)
    vo_ts1 = np.asarray(vo_ts1)

    # extract segment of traverse only if required

    if cutoffs is not None:
        # indices for slice
        l = cutoffs[0]
        u = cutoffs[1]
        # extract subset
        gt_ts1 = gt_ts1[l:u]
        gt_poses1 = gt_poses1[l:u]
        vo1 = vo1[l:u]
        vo_ts1 = vo_ts1[l:u]

    return gt_ts1, gt_poses1, vo_ts1, vo1


def read_global(traverse, tstamps, descriptor):
    dirpath = path.join(DATA_DIR, traverse, f'features/global/{descriptor}')
    glb_des = np.concatenate([np.load(path.join(dirpath, f'{ts}.npy'))
                              for ts in tstamps], axis=0)
    return glb_des


def read_local(traverse, tstamp, num_feats=None):
    with np.load(path.join(DATA_DIR, traverse, 'features/local',
                           str(tstamp) + ".npz")) as f:
        kp, des = preprocess_local_features(f, num_feats=num_feats)
    return kp, des


def read_local_raw(traverse, tstamp):
    with np.load(path.join(DATA_DIR, traverse, 'features/local',
                           str(tstamp) + ".npz")) as f:
        return dict(f)


def read_image(traverse, tstamp):
    img = cv2.imread(path.join(DATA_DIR, traverse,
                               'images/left', str(tstamp) + '.png'))
    return img


def preprocess_local_features(features, num_feats=None):
    des = features['local_descriptors'].astype(np.float32)[0]
    kp = features['keypoints'].astype(np.int32)
    scores = features['scores'].astype(np.float32)
    # keypoints must sorted according to score
    assert np.all(np.sort(scores)[::-1] == scores)
    # remove features detected on the car bonnet
    above_bonnet = kp[:, 1] < 800
    # filter features for ones detected above car bonnet
    kp = kp[above_bonnet, :]
    des = des[above_bonnet, :]
    scores = scores[above_bonnet]
    # retain based on num_feats if specified
    if num_feats is not None:
        kp = kp[:min(num_feats, len(kp)), :]
        des = des[:min(num_feats, len(des)), :]
    return kp, des


def gt_on_map_status(ref_gt, query_gt):
    '''For reference and query ground truth poses, decide if each query is within map or not'''
    query_gt_se2 = SE2(query_gt)
    ref_gt_se2 = SE2(ref_gt)
    min_dists_from_map = []
    for query_pose in query_gt_se2:
        xy_errs, rot_errs = (query_pose / ref_gt_se2).magnitude(degrees=False)
        closest_ref = np.argmin(xy_errs + 10. * rot_errs)
        min_dists_from_map.append((xy_errs[closest_ref], rot_errs[closest_ref]))
    min_xy_dist, min_rot_dist = np.asarray(min_dists_from_map).T
    gt_on_map_mask = np.logical_and(min_xy_dist < 5., min_rot_dist * 180. / np.pi < 30.)
    return gt_on_map_mask

#!/usr/bin/env python3
"""
A non-ROS script to visualize extracted keypoints and their matches of given image pairs
"""

import os
import cv2
import numpy as np
import time
import threading
import sys
from tqdm import tqdm
import argparse

from settings import DATA_DIR


def main(args):
    if args.net == 'hfnet_vino':
        from hfnet_vino import FeatureNet, default_config
    elif args.net == 'hfnet_tf':
        from hfnet_tf import FeatureNet, default_config
    else:
        exit('Unknown net %s' % args.net)

    # initialize network and load network weights
    config = default_config
    net = FeatureNet(config)

    # iterate through traverses and extract features
    pbar = tqdm(args.traverse)
    for traverse in pbar:
        pbar.set_description(traverse)
        # setup output dir for features
        local_dir = os.path.join(DATA_DIR, traverse, 'features/local')
        global_dir = os.path.join(DATA_DIR, traverse, 'features/global')
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(global_dir, exist_ok=True)

        d = os.path.join(DATA_DIR, traverse, 'images/left')
        d1 = os.path.join(DATA_DIR, traverse, 'images/right')
        fnames = [f for f in sorted(os.listdir(d)) if f.endswith(".png")]
        for f in tqdm(fnames, desc='images'):
            impath = os.path.join(d, f)
            image = cv2.imread(impath)
            if type(image) is not np.ndarray:
                # remove corrupt images from left and right cameras
                os.remove(impath)
                os.remove(os.path.join(d1, f))
                continue
            if args.net == 'hfnet_tf':
                image = cv2.resize(image, (960, 720))
            features = net.infer(image)

            # split local and global
            global_desc = features["global_descriptor"]
            features.pop("global_descriptor")  # local feature info only

            # save descriptors to disk
            np.save(os.path.join(global_dir, f[:-4]), global_desc)
            np.savez(os.path.join(local_dir, f[:-4]), **features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract hf-net features from images')
    parser.add_argument('--net', type=str, default='hfnet_vino',
        help='Network model: hfnet_vino (default), hfnet_tf.')
    parser.add_argument('--traverse', '-t', nargs='+', type=str, required=True, help='Traverse name to extract features from.')
    args = parser.parse_args()
    main(args)

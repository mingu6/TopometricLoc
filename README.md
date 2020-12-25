# Probabilistic Topometric Localization

Run the steps in the following order to get things working.

# Setup

## Initial setup

Clone repo, then run

```
cd TopometricLoc
sh setup.sh
```

This script will install the associated python package to this repository with dependencies. It will also download model weights for feature extraction (HF-Net with OpenVINO) and ask you to enter directories where data is stored (`DATA_DIR`) and results (`RESULTS_DIR`).

## Data

Data is zipped and ready on my desktop PC, come see me with a hard drive! On a high level, to ready the raw RobotCar traverses we need to:

- Name the traverse: date/time stamp is renamed to information in tags e.g. 2015-04-24-08-15-07 is renamed to sun_clouds_detour1. Original date/time stamp saved in 'decription.txt' in traverse folder.
- Undistort/debayerize images, see QVPR RobotCar repository 'ready_images.py' to do that. We only use stereo/left images (appearance only system)
- Align GPS (RTK) to camera poses using extrinsic transforms. Also interpolates ground truth GPS to camera timestamps. See QVPR RobotCar repository 'gps_camera_align.py' to do this.

Images are stored in `DATA_DIR/traverse_name/images/` and ground truth poses are stored in `DATA_DIR/traverse_name/camera_poses.csv`.

### To do:

- Automatic downloader script. I provide list of traverses and associated names, script downloads relevant data and saves it in the correct folder structure.

## Feature extraction

To extract features (local and global) from the undistored and debayerized images, use the `src/feature_extraction/extract_features.py` script. This will save the features with the images for use, with local and global features split into different folders and with each image having a corresponding feature file.

# Processing traverses

## Subsample traverses

After raw feature extraction and data processing of the entire traverse, we subsample the traverses based on odometry (given by VO) for mapping and localization. To do this, use the `src/data/subsample_traverse.py` script (use --help for information).

## Reference map building

Reference maps can be build from subsampled traverse data. Maps store the nodes with odometry constraints (segments) between them preprocessed before localization. Maps also store the global descriptors (NetVLAD from HF-Net) and timestamps (to load local descriptors from disk when required). This map object will be used frequently when localizing. To build a map, use the `src/mapping.py` script (see --help for information).

# Results

## Localization

Baselines are stored in the `src/baselines/` folder, and scripts include Localization objects which store state estimates, model parameters and can be iterated to update state estimates given odometry and appearance observations. Our method is stored in `src/localization.py`. Both the comparison methods and our method has the same class structure for `Localization` objects and are called in the `src/evaluate.py` script.

## Evaluation

Run `src/evaluate.py` to generate results. Script uniformly (spatially) samples the full query traverse as a starting point for global localization and runs each method (ours or comparisons) until convergence. It stores results in `RESULTS_DIR` with a description of the experiment which is automatically generated if none is provided (see --help for more information).

To do:

- Automatically tabulate and summarize results given a list of experiment names, export to latex for the paper

# Other

There is a folder `tests` with notebooks containing exploratory experiments. `tests/off_map_classifier-geom.ipynb` contains test for the off-map detector and allows you to play with parameters and test results on an on-map and off-map segment.
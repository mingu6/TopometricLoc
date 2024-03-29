# CODE IS NOT READY FOR RELEASE, WILL BE CLEANING UP IN COMING WEEKS

This repository contains code related to the following paper - please cite it if you use this code:
```bibtex
@article{xu2021probabilistic,
  title={Probabilistic Appearance-Invariant Topometric Localization with New Place Awareness},
  author={Xu, Ming and Fischer, Tobias and S{\"u}nderhauf, Niko and Milford, Michael},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={4},
  pages={6985--6992},
  year={2021}
}
```

# Probabilistic Topometric Localization

Run the steps in the following order to get things working.

# 1. Setup

## Initial setup

Clone repo, and setup virtual environment. OpenVINO only works with Python 3.7, so run

```bash
conda create env -n topometricloc python=3.7
conda activate topometricloc
```

Then run
```bash
cd TopometricLoc
sh setup.sh
```

This script will install the associated python package to this repository with dependencies. It will also download model weights for feature extraction (HF-Net with OpenVINO) and ask you to enter directories where data is stored (`DATA_DIR`) and results (`RESULTS_DIR`). Directories entered are stored in the `topometricloc/settings.py` file and used as global variables in scripts in this repo.

# 2. Data Format

To use this code for any dataset you like, simply adhere to the following data format. 

You will require a set of images with timestamps or frame order for filenames (sans file extension), corresponding global image descriptors for each image (e.g. NetVLAD, HF-Net), ground truth poses for each image (e.g. GPS) and finally odometry estimates between adjacent images.

The base directory for all data is the `DATA_DIR` directory. We assume the data is presented as a set of traverses, with each traverse occupying its own folder in `DATA_DIR`. An example valid directory structure is given as follows:

```txt
----
|-- DATA_DIR
|   |-- <traverse_1>
|   |   |-- images
|   |   |   |-- 0001.png
|   |   |   |-- ...
|   |   |   |-- 1000.png
|   |   |-- features
|   |   |   |-- 0001.npy
|   |   |   |-- ...
|   |   |   |-- 1000.npy
|   |   |-- camera_poses.csv
|   |   |-- odometry.csv
|   |-- ...
|   |-- <traverse_5>
|   |   |-- images
|   |   |   |-- 0001.png
|   |   |   |-- ...
|   |   |   |-- 0500.png
|   |   |-- features
|   |   |   |-- 0001.npy
|   |   |   |-- ...
|   |   |   |-- 0500.npy
|   |   |-- camera_poses.csv
|   |   |-- odometry.csv
```

### Raw Images

For a given traverse, raw images are stored in `DATA_DIR/<traverse_name>/images/` with arbitrary filename extensions. We also assume image names have a corresponding numeric (at least have the ability to be cast into an `int`!!) identifier which describes the order images are captured (e.g. timestamp). An example of a valid filename is given by `00001.png`.

### Image features/descriptors

Global features/descriptors are stored in `DATA_DIR/<traverse_name>/features/` as `.npy` files. Note, for a given traverse, each image in the `images/` folder MUST have a corresponding feature. For example, `00001.png` must have a corresponding feature `00001.npy` in the `features/` directory. Features are assumed to be stored as a 1D numpy array with shape `(D,)`, e.g. `(4096,)` for vanilla NetVLAD and HF-Net.

### Ground truth pose information

Ground truth poses for a trajectory must be stored in a single `.csv` file located at `DATA_DIR/<traverse_name>/camera_poses.csv`. The format of ground truth pose information is stored as a 6D pose with orientation given by a `r, p, y` Euler angle representation (please have mercy on my soul :p). All ground truth poses are given in the world coordinate frame as a world-to-body transform.

We store 6D poses for the purposes of applying one of our comparison methods (MCL) which requires 6DoF poses. If you have an alternative (lower) number of DoFs, e.g. 3, 4, then simply save a 6DoF pose with zeros in dimensions that are not used.

```txt
ts, x, y, z, r, p, y
0001, 1.0, 100.0, 0.5, 0.003, -0.06, 0.07
0002, 3.2, 105.0, 0.7, -0.01, -0.05, 0.075
...
```

### Odometry information

Odometry is defined as a relative pose between adjacent pairs `(source_frame, destination_frame)` of images and is given as a 6D relative pose. We assume the origin of the transformation is at the position of the source frame. As a simple check, composing the global pose of the source frame with the relative pose estimate between source and dest should yield the pose of the dest frame. Example: 

```txt
source_ts, dest_ts, x, y, z, r, p, y
0001, 0002, 1.0, 100.0, 0.5, 0.003, -0.06, 0.07
0002, 0003, 3.2, 105.0, 0.7, -0.01, -0.05, 0.075
...
```

Again, similar to ground truth poses, if odometry in a lower number of DoFs is provided, then fill in unused dimensions with zeros.

# 3. Feature extraction (CPU version of HF-Net)

We provide a helpful utility to easily extract features from images assuming the data structure in section 2 has been adhered to. The feature extraction method provided is an OpenVINO version of HF-Net for GPU-free feature extraction. Our code has minor changes to original code found in [this repo](https://github.com/cedrusx/deep_features).

To extract features from images, use the `topometricloc/feature_extraction/extract_features.py` script. You simply provide the folder name of the traverse you wish to extract features from located inside the `DATA_DIR` and it'll do it's thing!

# Processing traverses

## Subsample traverses

After raw feature extraction and data processing of the entire traverse, we subsample the traverses based on odometry (given by VO) for mapping and localization. To do this, use the `src/data/subsample_traverse.py` script (use `--help` for information).

## Reference map building

Reference maps can be build from subsampled traverse data. Maps store the nodes with odometry constraints (segments) between them preprocessed before localization. Maps also store the global descriptors (NetVLAD from HF-Net) and timestamps (to load local descriptors from disk when required). This map object will be used frequently when localizing. To build a map, use the `src/mapping.py` script (see `--help` for information).

# Results

## Localization

Baselines are stored in the `src/baselines/` folder, and scripts include Localization objects which store state estimates, model parameters and can be iterated to update state estimates given odometry and appearance observations. Our method is stored in `src/localization.py`. Both the comparison methods and our method has the same class structure for `Localization` objects and are called in the `src/evaluate.py` script.

## Evaluation

Run `src/evaluate.py` to generate results. Script uniformly (spatially) samples the full query traverse as a starting point for global localization and runs each method (ours or comparisons) until convergence. It stores results in `RESULTS_DIR` with a description of the experiment which is automatically generated if none is provided (see `--help` for more information).

Model parameters for each method are stored in the `src/params/` folder as yaml files.

`src/results.py` aggregates results into tables and outputs them as `.tex` files using pandas. The input to this script is a csv file storing the traverse/method/exper description information about the experiments to be aggregated.

# Other

There is a folder `tests` with notebooks containing exploratory experiments. `tests/off_map_classifier-geom.ipynb` contains a notebook for tuning the off-map detector parameters and allows you to change parameter values and evaluate detector performance on an on-map and off-map segment.

`src/visualization.py` allows you to visualize localization for our method for any traverse. Outputs a multitude of useful diagnostic plots to understand how the state estimate (belief) is being updated, where the state proposals are with confidence scores, sensor data (measurement likelihoods, motion, off-map detector, retrieved images). Very handy for tuning parameters on the training set!

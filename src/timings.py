import argparse
import os
import os.path as path
import timeit

import pickle
import yaml

from data.utils import load_pose_data, read_global, read_local
from mapping import RefMap
from localization import Localization
from settings import DATA_DIR, RESULTS_DIR

self_dirpath = os.path.dirname(os.path.abspath(__file__))

# feature extraction timing code
setup_feature = '''
import os
import numpy as np
import cv2
from settings import DATA_DIR
from feature_extraction.hfnet_vino import FeatureNet, default_config
from __main__ import q_traverse, tstampsQ
net = FeatureNet(default_config)
i = np.random.randint(0, len(tstampsQ))
d = os.path.join(DATA_DIR, q_traverse, 'images/left', str(tstampsQ[i]) + '.png')
img = cv2.imread(d)
'''

test_feature = '''
features = net.infer(img)
'''

# motion timing code

setup_motion = '''
from localization import Localization
import numpy as np
from __main__ import loc, odomQ
i = np.random.randint(0, len(odomQ))
'''

test_motion = '''
loc._update_motion(odomQ[i])
'''

# measurement (similarity) timing code

setup_sims = '''
import numpy as np
from __main__ import query_global, refMap
i = np.random.randint(0, len(query_global))
'''

test_sims = '''
query_sims = refMap.glb_des @ query_global[i]
'''

# measurement (retrieval) timing code

setup_retrieve = '''
import numpy as np
from measurement import retrieval_fn
from __main__ import query_global, refMap, params
i = np.random.randint(0, len(query_global))
query_sims = refMap.glb_des @ query_global[i]
meas = params['measurement']
'''

test_retrieve = '''
retrievals = retrieval_fn(query_sims, meas['k'], meas['smoothing_window'],
meas['smoothing_bandwidth'], meas['rho'], meas['alpha'])
'''

# measurement (verification) timing code

setup_verif = '''
import numpy as np
from data.utils import read_local
from measurement import geometric_verification, retrieval_fn, contiguous_peaks, peak_heights
from __main__ import query_global, refMap, params, q_traverse, tstampsQ
i = np.random.randint(0, len(query_global))

query_sims = refMap.glb_des @ query_global[i]
meas = params['measurement']
num_feats = meas['num_feats']

retrievals = retrieval_fn(query_sims, meas['k'], meas['smoothing_window'],
meas['smoothing_bandwidth'], meas['rho'], meas['alpha'])
peak_inds = contiguous_peaks(retrievals)
heights = sorted(enumerate(peak_heights(retrievals, peak_inds)),
                 key=lambda x: -x[1][1])[:meas['num_verif']]
verif_inds = [h[1][0] for h in heights]
temp_inds = np.asarray([h[0] for h in heights])  # indices of peak arr.
refLoc = [refMap.load_local(ind, num_feats) for ind in verif_inds]

qkp, qdes = read_local(q_traverse, tstampsQ[i], num_feats=num_feats)
inlier_threshold = meas['inlier_threshold']
num_inliers = meas['num_inliers']
confidence = meas['confidence']
verif_multiplier = meas['verif_multiplier']
'''

test_verif = '''
verif = [geometric_verification(kp, des, qkp, qdes,
                                num_inliers, inlier_threshold,
                                confidence) for kp, des in refLoc]
verif_succ = any(verif)
# find node index where verification was a success if any
try:
    success_ind = verif_inds[next(i for i, v in enumerate(verif) if v)]
except StopIteration:
    success_ind = None
# for successful verification, increase lhood of peak in retrievals
for inds in peak_inds[temp_inds[verif], :]:
    retrievals[inds[0]:inds[1]] *= verif_multiplier
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
    parser.add_argument("-d", "--description", type=str, default="",
                        help="description of model for experiment")
    parser.add_argument("-rt", "--reference-traverse", type=str, default="overcast1",
                        help="reference traverse name, e.g. overcast, night")
    parser.add_argument("-qt", "--query-traverse", type=str, default="night",
                        help="query traverse name, e.g. dusk, night")
    parser.add_argument("-rf", "--reference-filename", type=str, default='t_1_w_10_wd_3.pickle',
                    help="filename containing subsampled reference traverse poses")
    parser.add_argument("-qf", "--query-filename", type=str, default='t_1_w_10.csv',
                    help="filename containing subsampled query traverse poses")
    parser.add_argument("-p", "--params", type=str, default="",
                        help="filename containing model parameters")
    args = parser.parse_args()

    ref_traverse = args.reference_traverse
    q_traverse = args.query_traverse
    r_fname = args.reference_filename
    q_fname = args.query_filename

# read in parameters
    if not args.params:
        params_file = 'ours.yaml'
    else:
        params_file = args.params
    params_path = path.abspath(path.join(self_dirpath, "params"))
    with open(path.join(params_path, 'ours.yaml'), 'r') as f:
        params = yaml.safe_load(f)

    # load map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, r_fname)
    with open(fpath, "rb") as f:
        refMap = pickle.load(f)

    # load query sequence
    tstampsQ, xyzrpyQ, odomQ = load_pose_data(q_traverse, q_fname)
    query_global = read_global(q_traverse, tstampsQ)

    loc = Localization(params, refMap)

    # time for feature extraction
    num_feature = 1000
    feature_time = timeit.timeit(stmt=test_feature, setup=setup_feature,
                                 number=num_feature)
    print(f"feature extraction: average {feature_time * 1000 / num_feature:.2f}ms over {num_feature} trials")

    # time motion update for our method
    num_motion = 10000
    motion_time = timeit.timeit(stmt=test_motion, setup=setup_motion,
                                number=num_motion)
    print(f"motion update (ours): average {motion_time * 1000 / num_motion:.2f}ms over {num_motion} trials")

    # time image similarities
    num_sims = 10000
    sims_time = timeit.timeit(stmt=test_sims, setup=setup_sims,
                              number=num_sims)
    print(f"image similarities: average {sims_time * 1000 / num_sims:.2f}ms over {num_sims} trials")

    # time retrieval function eval 
    num_retrieve = 10000
    retrieve_time = timeit.timeit(stmt=test_retrieve, setup=setup_retrieve,
                                  number=num_retrieve)
    print(f"retrieval func (ours): average {retrieve_time * 1000 / num_retrieve:.2f}ms over {num_retrieve} trials")

    # time geometric verification 
    num_verif = 1000
    verif_time = timeit.timeit(stmt=test_verif, setup=setup_verif,
                                  number=num_verif)
    print(f"geom. verif.: average {verif_time * 1000 / num_verif:.2f}ms over {num_verif} trials")

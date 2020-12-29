import argparse
from copy import deepcopy
import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import sys

import cv2
import yaml

from data.utils import load_pose_data, read_global, read_local_raw, read_image
from mapping import RefMap
from measurement import retrieval_fn, off_map_detection
from motion import odom_deviation, transition_probs
from localization import Localization
from settings import DATA_DIR
from utils import pose_err

self_dirpath = os.path.dirname(os.path.abspath(__file__))


def vis_bayes_update(prior_belief, pred_belief, post_belief, refMap,
                     scores, query_gt, sInd, t, q_traverse):
    max_bel = max([prior_belief[:-1].max(), pred_belief[:-1].max(),
                   post_belief[:-1].max()])
    # first fig contains belief prob. superimposed on map
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

    # plot belief prior to motion update
    ax0_plt = ax[0].scatter(refMap.gt_poses[:, 1], refMap.gt_poses[:, 0],
                            c=prior_belief[:-1], cmap='YlOrRd',
                            vmin=0., vmax=max_bel)
    # add colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ax0_plt, cax=cax)
    # add title
    ax[0].set_title(f"Prior belief, off-map bel. {prior_belief[-1]:.2f}")
    # plot belief after motion update
    ax1_plt = ax[1].scatter(refMap.gt_poses[:, 1], refMap.gt_poses[:, 0],
                            c=pred_belief[:-1], cmap='YlOrRd',
                            vmin=0., vmax=max_bel)
    # add colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ax1_plt, cax=cax)
    # add title
    ax[1].set_title(f"Pred. belief (motion), off-map bel. {pred_belief[-1]:.2f}")
    # plot belief after measurement update
    ax2_plt = ax[2].scatter(refMap.gt_poses[:, 1], refMap.gt_poses[:, 0],
                            c=post_belief[:-1], cmap='YlOrRd',
                            vmin=0., vmax=max_bel)
    # add colorbar
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ax2_plt, cax=cax)
    # add title
    ax[2].set_title(f"Poster. belief (meas.), off-map bel. {post_belief[-1]:.2f}")
    # plot ground truth query pose to map
    for a in ax:
        a.scatter([query_gt[1]], [query_gt[0]], color='green', s=30,
                  label='query pose')
    # plot predicted position to map
    pred_ind = np.argmax(scores)
    ref_pred_gt = refMap.gt_poses[pred_ind]
    t_err, R_err = pose_err(ref_pred_gt, query_gt, degrees=True)
    for a in ax:
        a.scatter([ref_pred_gt[1]], [ref_pred_gt[0]], color='blue', s=30,
                  label='predicted node')
        a.annotate(f"score: {scores[pred_ind]:.2f}, t err: {t_err:.1f}m, {R_err:.0f}deg",
                   (ref_pred_gt[1] + 5, ref_pred_gt[0] + 5))
    # plot ground truth nearest position and confidence if within-map
    t_err, R_err = pose_err(query_gt, refMap.gt_poses, degrees=False)
    tot_err = t_err + 5. * R_err
    ind_gt = np.argmin(tot_err)
    if tot_err.min() <= 10.:  # if camera is within the map, has nearby ref.
        ref_near_gt = refMap.gt_poses[ind_gt]
        for a in ax:
            a.scatter([ref_near_gt[1]], [ref_near_gt[0]], color='pink', s=30,
                      label='nearest ref.')
            a.annotate(f"{scores[pred_ind]:.2f}",
                       (ref_near_gt[1] + 5, ref_near_gt[0] + 5))
            a.scatter([query_gt[1]], [query_gt[0]], color='green', s=30,
                      label='true query')
    # formatting of subplots
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('equal')
        a.legend()
    # make fig larger
    old_fig_size = fig.get_size_inches()
    fig.set_size_inches(old_fig_size[0] * 4.0, old_fig_size[1] * 2.0)
    fig.suptitle(f"traverse: {q_traverse}, start ind.: {sInd},"
                 f" t: {t}, query ind.: {sInd+t}", fontsize=20)
    fig.tight_layout()

    # second fig contains belief expressed by place as a 1D plot

    fig1, ax1 = plt.subplots(1, 3, sharey=True, sharex=True)
    ax1[0].plot(prior_belief[:-1])
    ax1[0].set_title(f"Prior belief, off-map bel. {prior_belief[-1]:.2f}")
    ax1[0].set_ylabel("Belief", rotation=90)
    # plot nearest ref. node if query within map
    if tot_err.min() <= 10.:
        ax1[0].scatter([ind_gt], [prior_belief[ind_gt]], color="pink",
                       label="nearest ref.", s=60)
    # plot pred. node
    ax1[0].scatter([pred_ind], [prior_belief[pred_ind]], color="blue",
                   label="pred. ref.", s=60)
    ax1[1].plot(pred_belief[:-1])
    ax1[1].set_title(f"Pred. belief (motion), off-map bel. {pred_belief[-1]:.2f}")
    # plot nearest ref. node if query within map
    if tot_err.min() <= 10.:
        ax1[1].scatter([ind_gt], [pred_belief[ind_gt]], color="pink",
                       label="nearest ref.", s=60)
    # plot pred. node
    ax1[1].scatter([pred_ind], [pred_belief[pred_ind]], color="blue",
                   label="pred. ref.", s=60)
    ax1[2].plot(post_belief[:-1])
    ax1[2].set_title(f"Poster. belief (meas.), off-map bel. {post_belief[-1]:.2f}")
    # plot nearest ref. node if query within map
    if tot_err.min() <= 10.:
        ax1[2].scatter([ind_gt], [post_belief[ind_gt]], color="pink",
                       label="nearest ref.", s=60)
    # plot pred. node
    ax1[2].scatter([pred_ind], [post_belief[pred_ind]], color="blue",
                   label="pred. ref.", s=60)
    for a in ax1:
        a.set_xlabel("Reference Nodes/Places")  # xlabels
        a.legend()
    # plot predicted index
    old_fig_size1 = fig1.get_size_inches()
    fig1.set_size_inches(old_fig_size1[0] * 4.0, old_fig_size1[1] * 1.5)
    fig1.suptitle("1D belief plots")
    fig1.tight_layout()

    return None


def vis_sensor(query_img, query_sims, meas_lhood, on_success, on_to_off,
               k, refMap, query_gt):
    # first plot contains model outputs from meas. and motion updates

    if on_to_off is not None:
        fig, ax = plt.subplots(1, 3)
    else:
        fig, ax = plt.subplots(1, 2)  # t=0 has no motion update, dont visualize

    # plot measurement likelihood
    ind_retrieved = np.argsort(-query_sims)[:k]
    ax[0].plot(meas_lhood)
    ax[0].set_ylim(0., meas_lhood.max() + 0.5)
    ax[0].scatter(ind_retrieved, meas_lhood[ind_retrieved], color='red',
                  label='retrieved pts')
    ax[0].set_xlabel('Reference nodes/places')
    ax[0].set_title(f'Measurement likelihood, on verified: {on_success}')
    # plot top-k retrievals and normalized similarities
    sims_retrieved = query_sims[ind_retrieved]
    sims_min = sims_retrieved.min()
    sims_norm = sims_retrieved - sims_min
    norm_const = sims_norm.sum()
    sims_norm /= norm_const
    ax[1].bar(ind_retrieved, sims_norm, width=20, color='purple',
              label='normalized similarities')
    ax[1].set_xlim(0, len(meas_lhood))
    ax[1].set_xlabel('Reference nodes/places')
    # plot off-map transition prob. for each place
    if on_to_off is not None:
        ax[2].plot(on_to_off, label='node to off-map transition prob.')
        ax[2].set_ylim(0., 1.)
        ax[2].set_ylabel('off-map transition prob.', rotation=90)
        ax[2].set_xlabel('Reference nodes/places')
    # plot gt index and positions
    t_err, R_err = pose_err(query_gt, refMap.gt_poses, degrees=False)
    tot_err = t_err + 5. * R_err
    ind_gt = np.argmin(tot_err)
    if tot_err.min() <= 10.:  # if camera is within the map, has nearby ref.
        ax[0].scatter([ind_gt], [meas_lhood[ind_gt]], color='pink', s=30,
                      label='nearest ref.')
        sims_norm_gt = (query_sims[ind_gt] - sims_min) / norm_const
        ax[1].bar([ind_gt], [sims_norm_gt], color='pink', width=50,
                  label='nearest ref.')
        if on_to_off is not None:
            ax[2].scatter([ind_gt], [on_to_off[ind_gt]], color='pink', s=30,
                          label='nearest ref.')
    # formatting
    for a in ax:
        a.legend()
    old_fig_size = fig.get_size_inches()
    fig.set_size_inches(old_fig_size[0] * 4.0, old_fig_size[1] * 1.0)
    fig.tight_layout()

    # plot query img with top-k retrieved images

    wd = int(np.sqrt(k+1)) + 1
    fig1, ax1 = plt.subplots(wd, wd)
    ax1[0, 0].imshow(cv2.resize(query_img, (320, 240)))
    ax1[0, 0].set_title('Query Image')
    for i in range(wd):
        for j in range(wd):
            ax1[i, j].set_xticks([])
            ax1[i, j].set_yticks([])
    for i in range(k+1):
        if i > 0:
            row = int(i / wd)
            col = i % wd
            # load retrieved ref. image
            ref_img = cv2.resize(refMap.load_image(ind_retrieved[i-1]),
                                 (320, 240))
            # compute pose err. from query
            ref_gt = refMap.gt_poses[ind_retrieved[i-1]]
            t_err, R_err = pose_err(ref_gt, query_gt, degrees=True)
            ax1[row, col].imshow(ref_img)
            ax1[row, col].set_title(f"t err.: {t_err:.1f}m, R err.: {R_err:.0f}")

    old_fig_size1 = fig1.get_size_inches()
    fig1.set_size_inches(old_fig_size1[0] * k  / 12., old_fig_size1[1] * k / 12.)
    fig1.tight_layout()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build topological map from subsampled traverse data"))
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
    parser.add_argument("-t", "--trials", type=int, default=1000,
                        help="number of trials to run, evenly distributed")
    args = parser.parse_args()

    ref_traverse = args.reference_traverse
    q_traverse = args.query_traverse
    r_fname = args.reference_filename
    q_fname = args.query_filename

    if args.params:
        params_file = args.params
    else:
        params_file = 'ours.yaml'

    # read in parameters
    params_path = path.abspath(path.join(self_dirpath, "params"))
    with open(path.join(params_path, params_file), 'r') as f:
        params = yaml.safe_load(f)
    meas_params = params['measurement']
    motion_params = params['motion']
    other_params = params['other']

    # load map
    map_dir = path.join(DATA_DIR, ref_traverse, 'saved_maps')
    fpath = path.join(map_dir, r_fname)
    with open(fpath, "rb") as f:
        refMap = pickle.load(f)

    # load query sequence
    tstampsQ, xyzrpyQ, odomQ = load_pose_data(q_traverse, q_fname)
    query_global = read_global(q_traverse, tstampsQ)
    # compute distance travelled (meters) from starting location
    relpose = xyzrpyQ[1:, :2] - xyzrpyQ[:-1, :2]
    relpose = np.vstack((np.zeros(2), relpose))
    dist_from_start = np.cumsum(np.linalg.norm(relpose, axis=1))

    # subsample query traverse for starting point
    spacing = int(len(tstampsQ) / args.trials)  # gap between start of next trial
    start_inds = np.arange(0, len(tstampsQ) - 50, spacing)  # last trial has space

    # enter trial number to visualize
    while True:
        user = input(f"Enter trial number /{len(start_inds)} (type q to quit): ")
        try:
            trial_no = int(user)
            if trial_no not in range(len(start_inds) - 1):
                print(f"Invalid value {user}! Enter an integer between"
                      f"0 and {len(start_inds) - 1} or 'q' to quit.")
                continue
            else:
                sInd = start_inds[trial_no]
        except ValueError as e:
            if user == 'q':
                sys.exit("Exiting...")
            else:
                print(e)
                print(f"Invalid value {user}! Enter an integer between 0"
                      f"and {len(start_inds) - 1} or 'q' to quit.")
                continue

        # store output from localization
        belief_prior = []  # belief before motion and meas. update
        belief_pred = []  # belief after motion update
        belief_post = []  # belief after motion and meas. update
        on_detections = []  # off/on map detection at each step
        meas_lhood = []  # measurement likelihood at each step
        on_to_off = [None]  # state transitions off-map at each step (none at t=0)
        img_retrievals = []  # location of image retrievals at each step
        query_sims_all = []  # query similarities
        scores_all = []  # convergence scores
        gt_errs = []

        # run localization until convergence
        loc = Localization(params, refMap)

        for t, s in enumerate(range(sInd, len(odomQ))):
            qLoc = read_local_raw(q_traverse, tstampsQ[s])
            # usually at t=0 there is a meas. update with no motion
            # separate initialization performed
            if t == 0:
                belief_prior.append(loc.belief.copy())
                belief_pred.append(loc.belief.copy())  # no motion update at t=0
                loc.init(odomQ[s], query_global[s], qLoc)
                belief_post.append(loc.belief.copy())
            else:
                # update state estimate
                belief_prior.append(loc.belief.copy())
                loc._update_motion(odomQ[s-1])
                belief_pred.append(loc.belief.copy())
                loc._update_meas(query_global[s], qLoc)
                belief_post.append(loc.belief.copy())
                # image retrieval
                query_sims = refMap.glb_des @ query_global[s]
                query_sims_all.append(query_sims)
                retrieval_inds = np.argsort(-query_sims)[:meas_params['k']]
                img_retrievals.append(retrieval_inds)
                # off-map detection and meas. lhood
                r = retrieval_fn(query_sims, meas_params['k'],
                                 meas_params['smoothing_window'],
                                 meas_params['smoothing_bandwidth'],
                                 meas_params['rho'], meas_params['alpha'])
                r_updated, on_detected, _ = off_map_detection(
                    qLoc, refMap, r, meas_params['num_feats'],
                    meas_params['num_verif'], meas_params['verif_multiplier'],
                    meas_params['num_inliers'], meas_params['inlier_threshold'],
                    meas_params['confidence'])
                meas_lhood.append(r_updated + 1.)  # store meas. lhood
                on_detections.append(on_detected)
                # compute transition probs
                att_wt = motion_params["att_wt"]
                # compute deviations and within -> within/off probabilities
                dev = odom_deviation(odomQ[s-1], loc.odom_segments, att_wt)
                within, off_probs = transition_probs(
                    dev, motion_params["p_off_min"], motion_params["p_off_max"],
                    motion_params["d_min"], motion_params["d_max"],
                    motion_params["theta"])
                on_to_off.append(off_probs)

            # check convergence
            ind_max, converged, scores = loc.converged(
                other_params['convergence_score'],
                other_params['convergence_window'])
            scores_all.append(scores)
            # evaluation against ground truth
            t_err, R_err = pose_err(xyzrpyQ[sInd+t], refMap.gt_poses[ind_max])
            gt_errs.append((t_err, R_err))
            if converged:
                break

        # display sequence info
        steps_to_loc = t
        print(f"Convergence: {t} steps, meters travelled: "
              f"{dist_from_start[s] - dist_from_start[sInd]:.1f}m")

        while True:
            t_select = input(f"Enter a timestep (0-{steps_to_loc-1}) to visualize "
                             "(press 'b' to go back to trial selection, "
                             "'q' to exit): ")
            try:
                t_select = int(t_select)
                if t_select not in range(steps_to_loc):
                    print(f"{t_select} is invalid, please select an integer "
                          f"from 0-{steps_to_loc-1}")
                    continue
                else:
                    # successfully selected timestep, plot
                    vis_bayes_update(belief_prior[t_select],
                                     belief_pred[t_select], belief_post[t_select],
                                     refMap, scores_all[t_select],
                                     xyzrpyQ[sInd+t_select],
                                     sInd, t_select, q_traverse)
                    # import query img
                    query_img = read_image(q_traverse, tstampsQ[sInd+t_select])
                    vis_sensor(query_img, query_sims_all[t_select],
                               meas_lhood[t_select], on_detections[t_select],
                               on_to_off[t_select], meas_params['k'],
                               refMap, xyzrpyQ[sInd+t_select])
                    plt.show()
            except ValueError as e:
                if t_select == 'b':
                    print("Going back to trial selection...")
                    break
                elif t_select == 'q':
                    print("Exiting...")
                    sys.exit()
                else:
                    print(e)
                    print(f"{t_select} is invalid, please select an integer "
                          f"from 0-{steps_to_loc-1}")

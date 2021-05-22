import os.path as path
import numpy as np

from ..settings import RESULTS_DIR

# convergence score thresholds, more granular near 1
convergence_score_thresh = np.hstack((np.linspace(0., 0.9, 45, endpoint=False),
                                      np.linspace(0.9, 1., 100, endpoint=False)))

colors = {"Ours": "green",
          "No Off": "purple",
          "Baseline": "blue",
          "Xu20": "red",
          "Stenborg20": "orange"}


def load_results(fname, exper='loop_closure', suffix=''):
    """
    Load saved results from wakeup and loop closure experiments. Returns None if no file.
    """
    fpath = path.join(RESULTS_DIR, exper, fname, f"results{suffix}.npz")
    try:
        results = np.load(fpath)
    except FileNotFoundError as e:
        print(e)
        results = None
    return results


def tstep_first_converged(converged_threshold, convergence_scores):
    '''
    For each trial identify first timestep where convergence score
    exceeds desired threshold. For wakeup task only, not loop closure.
    Args:
        converged_threshold ndarray len m: vector of desired thresholds
        convergence_scores ndarray n_trials x n_steps: convergence scores for each trial at each timestep
    Returns:
        timestep_converged ndarray m x n_trials: array containing for each
            convergence threshold which timestep the trial converged. For
            trials which failed to converge at a given threshold, set -1
    '''
    assert converged_threshold.ndim == 1
    assert convergence_scores.ndim == 2
    above_threshold = convergence_scores > converged_threshold[:, None, None]
    timestep_converged = above_threshold.argmax(axis=2)  # returns index of first True
    failed_to_converge = np.logical_not(np.any(above_threshold, axis=2))
    timestep_converged[failed_to_converge] = -1
    return timestep_converged


def check_gt_err_within_tol(xy_err, rot_err, xy_tol, rot_tol):
    '''
    Check if ground truth error of prediction from model is within desired
    tolerances. Returns boolean True/False flags indicating within/outisde tol.
    Args:
        xy_err ndarray: array containing translation error (m) from model prediction
        rot_err ndarray: array containing rotation error (deg) from model prediction
        xy_tol float: max translation error (m) to be considered localization success
        rot_tol float: max rotation error (deg) to be considered localization success
    Returns:
        prediction_within_tol ndarray: Mask of timesteps where model prediction falls inside tolerance
    '''
    assert xy_err.shape == rot_err.shape
    prediction_within_tol = np.logical_and(xy_err < xy_tol, rot_err < rot_tol)
    return prediction_within_tol


def determine_confusion_status(converged_mask, within_gt_tol, on_map_gt_mask):
    '''
    Determines confusion status (i.e. TP, FP, TN, FN) for each convergence score
    threshold and prediction unit (e.g. single place in LCD or full trial in wakeup). Each status is given a key:
    TP = 0: Robot is within map and localization converges to correct place
    FP = 1: Localization converges to the incorrect place
    TN = 2: Robot is off-map and fails to converge
    FN = 3: Robot fails to converge but is within the map at end of trial
    Args:
        converged_mask ndarray m x n_pred: whether or not convergence criterion met for given score/prediction.
        within_gt_tol ndarray m x n_pred: whether or not gt error tolerance attained for score/prediction combo.
        on_map_gt_mask ndarray m x n_pred: whether or not robot was on-map during prediction.
    Returns:
        confusion_status ndarray m x n_pred: confusion status for each score/trial combination encoded using key
    '''
    m, n_pred = converged_mask.shape
    confusion_status = np.empty((m, n_pred), dtype=int)
    # Determine TP, must converge and be within gt err tolerance
    converged_within_tol = np.logical_and(within_gt_tol, converged_mask)
    confusion_status[converged_within_tol] = 0
    # Determine FP, must converge but fail to be within gt err tolerance
    converged_outside_tol = np.logical_and(np.logical_not(within_gt_tol), converged_mask)
    confusion_status[converged_outside_tol] = 1
    # Determine TN, must be off map and fails to converge
    failed_while_off = np.logical_not(np.logical_or(converged_mask, on_map_gt_mask))
    confusion_status[failed_while_off] = 2
    # Determine FN, must be within map but fails to converge
    failed_while_on = np.logical_and(on_map_gt_mask, np.logical_not(converged_mask))
    confusion_status[failed_while_on] = 3
    # Final check that confusion matrix is fully populated
    mask = converged_within_tol + converged_outside_tol + failed_while_off + failed_while_on
    assert np.all(mask), "confusion matrix not fully populated"
    return confusion_status


def precision_recall(confusion_status):
    '''
    Computes precision and recall given confusion status matrix.
    Args:
        confusion_status ndarray m x n_trials: confusion_status for each convergence score/trial combination.
    Returns:
        precision ndarray len m: precision for each convergence score threshold
        recall ndarray len m: recall for each convergence score threshold
    '''
    num_TP = (confusion_status == 0).sum(axis=1)
    num_FP = (confusion_status == 1).sum(axis=1)
    num_FN = (confusion_status == 3).sum(axis=1)
    precision = num_TP / (num_TP + num_FP)
    recall = num_TP / (num_TP + num_FN)
    precision[np.isnan(precision)] = 1.  # no predictions made
    return precision, recall

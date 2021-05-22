import numpy as np

class LocalizationBase:
    '''
    Base class for all methods. Must implement provided methods.
    '''
    def __init__(self, params, ref_map):
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]
        self.ref_map = ref_map
        self.belief = None

    def init(self, odom_mu, odom_sigma, global_desc):
        meas_lhood = None
        raise NotImplementedError
        return meas_lhood

    def _update_motion(self, odom_mu, odom_sigma):
        trans_mat = None
        raise NotImplementedError
        return trans_mat

    def _update_meas(self, global_desc):
        meas_lhood = None
        raise NotImplementedError
        return meas_lhood

    def update(self, odom_mu, odom_sigma, global_desc):
        self._update_motion(odom_mu, odom_sigma)
        self._update_meas(global_desc)
        return None

    def converged(self, belief=None):
        """
        Convergence detection for belief. Returns pose estimate and convergence score. Overwrite for non-discrete filters or alternative methods.
        Args:
            belief (optional): Belief vector of map states. If none, uses belief stored in object.
        Returns:
            pose_est: Proposed robot pose estimate. Inherits ground truth pose of selected place.
            score: Convergence score representing concentration of belief around a single place.
        """
        if belief is None:
            belief = self.belief
        window = self.other_params['convergence_window']
        sum_belief = np.convolve(belief[:-1], np.ones(2 * window + 1), mode='same')
        proposed_map_place = np.argmax(belief[:-1])
        score = sum_belief[proposed_map_place]
        pose_estimate = self.ref_map.gt_poses[proposed_map_place]
        return pose_estimate, score

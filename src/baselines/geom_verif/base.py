from measurement import retrieval_fn, off_map_detection

class Localization:
    def __init__(self, params, refMap):
        # model parameters
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]

        # reference map
        self.refMap = refMap

        self.predict_ind = None  # estimated state index
        self.verified = False  # flag changes if verification succeeds

    def init(self, qOdom, qGlb, qLoc):
        """
        Allows for any initialization at time 0 before first motion update
        """
        self.update(qOdom, qGlb, qLoc)
        return None

    def update(self, qOdom, qGlb, qLoc):
        """
        Updates most likely state estimate by geom. verif.
        """
        query_sims = self.refMap.glb_des @ qGlb
        # retrieval fn for within-map update and geom. verif.
        r = retrieval_fn(query_sims, self.meas_params['k'],
                         self.meas_params['smoothing_window'],
                         self.meas_params['smoothing_bandwidth'],
                         self.meas_params['rho'], self.meas_params['alpha'])
        # performs geometric verification for top few peaks in retrievals
        retrievals, on_detected, i = off_map_detection(
            qLoc, self.refMap, r, self.meas_params['num_feats'],
            self.meas_params['num_verif'], self.meas_params['verif_multiplier'],
            self.meas_params['num_inliers'], self.meas_params['inlier_threshold'],
            self.meas_params['confidence']
        )
        if on_detected:
            self.predict_ind = i
            self.verified = True
        return None

    def converged(self, score_thresh, nhood_size):
        localized = self.verified
        ind_max = self.predict_ind
        return ind_max, localized, None

import numpy as np

from ours.measurement import off_map_detection


class OnlineLocalization:
    def __init__(self, params, refMap):
        # model parameters
        self.meas_params = params["measurement"]

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
        meas_params = self.meas_params
        query_sims = self.refMap.glb_des @ qGlb
        # performs geometric verification for top few peaks in retrievals
        on_detected, i = off_map_detection(
            qLoc, self.refMap, query_sims, meas_params['num_feats'],
            meas_params['window'], meas_params['num_verif'],
            meas_params['num_inliers'], meas_params['inlier_threshold'],
            meas_params['confidence']
        )
        if i is not None:
            self.predict_ind = i
        else:
            self.predict_ind = np.argmax(query_sims)
        if on_detected:
            self.verified = True
        return None

    def converged(self, qGlb, qLoc):
        localized = self.verified
        ind_max = self.predict_ind
        score = 1. if localized else 0.
        return ind_max, localized, score

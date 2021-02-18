import numpy as np

import scipy.sparse as sparse


def create_transition_matrix(N, lower, upper):
    """
    Create state transition matrix with uniform transition probabilities.
    Args:
        N: Number of states/places
        lower: index of earliest relative transition. e.g. -2 means given a
               state, a transition of -2 nodes is possible.
        upper: index of furthest relative transition
    """
    assert upper > lower
    data = np.ones((upper - lower, N)) / (upper - lower)
    mat = sparse.diags(data, offsets=np.arange(lower, upper), shape=(N, N),
                       format="csr", dtype=np.float32)
    return mat.tocsc()


class OnlineLocalization:
    def __init__(self, params, refMap):
        # model parameters
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]

        # initialize belief
        N = refMap.N
        self.belief = np.ones(N) / N

        # measurement param
        self.delta = self.meas_params['delta']
        self.lambd = None

        # motion transition matrix
        self.E = create_transition_matrix(N, self.motion_params['lower'],
                                          self.motion_params['upper'])

        # reference map
        self.refMap = refMap

        # reset parameters
        self.reset = params["other"]["reset_step"]
        self.t = 0

    def init(self, qOdom, qGlb, qLoc, uniform=False):
        """
        Allows for any initialization at time 0 before first motion update.
        Calibrates intensity parameter lambda using first VPR measurement
        and sets belief. Option to set a uniform belief initialization.
        """
        query_sims = self.refMap.glb_des @ qGlb
        dist = np.sqrt(2. - 2. * query_sims)
        descriptor_quantiles = np.quantile(dist, [0.025, 0.975])
        quantile_range = descriptor_quantiles[1] - descriptor_quantiles[0]
        self.lambd = np.log(self.delta) / quantile_range
        lhood = np.exp(-self.lambd * dist)
        if not uniform:
            self.belief = lhood
            self.belief /= self.belief.sum()
        else:
            self.belief = np.ones(self.refMap.N) / self.refMap.N
        return lhood

    def _update_motion(self, qOdom):
        """
        Applies motion update to belief.
        """
        self.belief = self.E.T @ self.belief
        return None

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        # compute likelihoods
        query_sims = self.refMap.glb_des @ qGlb
        dist = np.sqrt(2. - 2. * query_sims)
        lhood = np.exp(-self.lambd * dist)
        self.belief *= lhood
        self.belief /= self.belief.sum()
        return lhood

    def update(self, qOdom, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        # reset if not converged by step limit
        if self.t >= self.reset:
            self.t = 0
            self.init(qOdom, qGlb, qLoc, uniform=True)
        # state update
        self._update_motion(qOdom)
        self._update_meas(qGlb, qLoc)
        # update step count since last reset
        self.t += 1
        return None

    def converged(self, qGlb, qLoc):
        window = self.other_params['convergence_window']
        score_thres = self.other_params['convergence_score']

        # take window around posterior mode, check prob. mass underneath
        sum_belief = np.convolve(self.belief, np.ones(2 * window + 1),
                                 mode='same')
        ind_max = np.argmax(sum_belief)
        ind_max = np.argmax(self.belief)
        score = sum_belief[ind_max]
        localized = score > score_thres

        # ind_max = np.argmax(self.belief)
        # nhood_inds = np.arange(max(ind_max-window, 0),
                               # min(ind_max+window, len(self.belief)))
        # belief_nhood = self.belief[nhood_inds]
        # localized = belief_nhood.sum() > score_thres
        # ind_pred = round(nhood_inds.mean())

        # # check that only one mode exists, identify next largest mode

        # belief_alt = self.belief[:-1].copy()
        # larger_window = np.arange(max(ind_pred-window*3, 0),
                                  # min(ind_pred+window*3, len(self.belief)-1))
        # belief_alt[larger_window] = 0.
        # ind_max_next = np.argmax(belief_alt)
        # nhood_inds_next = np.arange(max(ind_max_next-window, 0),
                                    # min(ind_max_next+window, len(self.belief)-1))
        # belief_nhood_next = belief_alt[nhood_inds_next]

        # # if second mode exists (with meaningful mass), set 0 score so model
        # # does not converge upon computing curves used in results

        # score = belief_nhood.sum()
        # if belief_nhood_next.sum() > 0.05:
            # localized = False
            # score = 0.

        return ind_max, localized, score

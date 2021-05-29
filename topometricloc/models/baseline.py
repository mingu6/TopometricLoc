import numpy as np

from .localization_base import LocalizationBase


class Localization(LocalizationBase):
    def __init__(self, params, ref_map):
        super().__init__(params, ref_map)

    def init(self, global_desc):
        self._update_meas(global_desc)
        return None

    def _update_motion(self, odom_mu, odom_sigma):
        return None

    def _update_meas(self, global_desc):
        query_sims = self.ref_map.glb_des @ global_desc
        dist = np.sqrt(2. - 2. * query_sims)
        self.belief = -dist
        return None

    def check_convergence(self):
        proposed_map_place = np.argmax(self.belief)
        score = self.belief[proposed_map_place]
        pose_estimate = self.ref_map.gt_poses[proposed_map_place]
        return pose_estimate, score

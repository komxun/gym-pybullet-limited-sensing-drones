"""Thin wrapper around AutoroutingSARLAviary with YAML-configurable reward parameters.

All critical bugs have been fixed directly in the parent classes:
- ExtendedSARLAviary: action timing, obs normalization, obs shape (1D)
- AutoroutingSARLAviary: terminated/truncated, vel_z normalization, reward (additive safety)
- BaseRouting: _resetSpeedCommand enum fix
- FCDuelingQ / FCQ: experience tuple order fix

This wrapper only adds YAML-configurable reward parameters on top of the fixed parents.
If no config is provided, it behaves identically to AutoroutingSARLAviary.
"""

import numpy as np
from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class CollisionAvoidanceAviary(AutoroutingSARLAviary):
    """AutoroutingSARLAviary with YAML-configurable reward parameters.

    Inherits all physics, routing, rendering, and bug fixes from parent classes.
    Only overrides _computeReward to use configurable parameters from a Config object.
    """

    def __init__(self, cfg=None, **kwargs):
        """
        Parameters
        ----------
        cfg : Config, optional
            If provided, reward parameters are read from cfg.
            Otherwise falls back to parent defaults (hardcoded values).
        **kwargs
            Passed through to AutoroutingSARLAviary.
        """
        self._cfg = cfg

        # Store reward config (with defaults matching parent's hardcoded values)
        if cfg is not None:
            self._reach_threshold = cfg.reward.reach_threshold_m
            self._reach_reward = cfg.reward.reach_reward
            self._collision_penalty = cfg.reward.collision_penalty
            self._time_penalty = cfg.reward.time_penalty
            self._safety_scale = cfg.reward.safety_penalty_scale
            self._progress_weight = cfg.reward.progress_weight
            self._additive_safety = cfg.reward.additive_safety
        else:
            self._reach_threshold = 1.0
            self._reach_reward = 10.0
            self._collision_penalty = -10.0
            self._time_penalty = -0.01
            self._safety_scale = 1.0
            self._progress_weight = 1.0
            self._additive_safety = True

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Configurable reward function (uses YAML params instead of hardcoded)
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Compute reward using configurable parameters from YAML config.

        Same structure as the fixed parent reward, but with tunable:
        - reach_threshold, reach_reward, collision_penalty
        - time_penalty, safety_penalty_scale, progress_weight
        - additive_safety (True = add penalty to progress, False = replace)
        """
        state = self._getDroneStateVector(0)
        cur_pos = np.array(state[0:3])

        prev_d2dest = np.linalg.norm(self.routing[0].DESTINATION - self.routing[0].CUR_POS)
        cur_d2dest = np.linalg.norm(self.routing[0].DESTINATION - state[0:3])
        self.routing[0].CUR_POS = cur_pos

        # --- Progress reward (potential-based shaping) ---
        progress = (prev_d2dest - cur_d2dest) * self._progress_weight

        # --- Time penalty ---
        reward = progress + self._time_penalty

        # --- Safety penalty (proximity to obstacles/other drones) ---
        detected_ratios = self.routing[0].RAYS_INFO[:, 0]
        safe_ratio = self.routing[0].ROV / self.routing[0].RAY_LEN_M
        nonzero_ratios = detected_ratios[np.nonzero(detected_ratios)]

        if len(nonzero_ratios) > 0 and np.any(nonzero_ratios < safe_ratio):
            min_ratio = np.min(nonzero_ratios)
            penalty = -self._safety_scale * ((safe_ratio - min_ratio) / safe_ratio) ** 2
            if self._additive_safety:
                reward += penalty
            else:
                reward = penalty

        # --- Terminal conditions (override everything) ---
        if cur_d2dest < self._reach_threshold:
            reward = self._reach_reward
        elif int(self.CONTACT_FLAGS[0]) == 1:
            reward = self._collision_penalty

        self.CUM_REWARD += reward
        return reward

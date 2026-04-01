import numpy as np

from gym_pybullet_drones.envs.ExtendedSARLAviary import ExtendedSARLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, RouteStatus

class AutoroutingSARLAviary(ExtendedSARLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.HB,
                 num_drones: int=1,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 30,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.AUTOROUTING,
                 sensor_cfg: dict=None,
                 action_cfg: dict=None,
                 skip_drone_raycasting: bool=False,
                 obs_choice: str="sensor",
                 episode_len_sec: int=30,
                 mission_cfg: dict=None
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        sensor_cfg : dict, optional
            Sensor parameters passed through to routing classes.
        action_cfg : dict, optional
            Action parameters (accel_value, decel_value) for the agent drone.

        """
        # self.TARGET_POS = np.array([0.2, 8, 1])
        self.EPISODE_LEN_SEC = episode_len_sec
        self.CUM_REWARD = 0
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         sensor_cfg=sensor_cfg,
                         action_cfg=action_cfg,
                         skip_drone_raycasting=skip_drone_raycasting,
                         obs_choice=obs_choice,
                         mission_cfg=mission_cfg
                         )
        self.CURRENT_POS = self.HOME_POS

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """
        state = self._getDroneStateVector(0)
        curPos = np.array(state[0:3])
        
        norm_ep_time = (self.step_counter/self.PYB_FREQ ) / self.EPISODE_LEN_SEC
        elapsed_time_sec = self.step_counter/self.PYB_FREQ

        # ---------Reward design-------------
        reachThreshold_m = 1  #0.2
        reward_choice = 3  # 8: best  10: 2nd best 11: Good
        # prevd2destin = np.linalg.norm(self.TARGET_POS - self.CURRENT_POS)
        # d2destin = np.linalg.norm(self.TARGET_POS - state[0:3])
        # h2destin = np.linalg.norm(self.TARGET_POS - self.HOME_POS)
        # self.CURRENT_POS = curPos
        # CHECK THIS AGAIN!!!!!!
        prevd2destin = np.linalg.norm(self.routing[0].DESTINATION - self.routing[0].CUR_POS)
        d2destin = np.linalg.norm(self.routing[0].DESTINATION - state[0:3])
        h2destin = np.linalg.norm(self.routing[0].DESTINATION - self.routing[0].HOME_POS)
        self.routing[0].CUR_POS = curPos

        detected_ratios = self.routing[0].RAYS_INFO[:,0]
        safe_detected_ratio = self.routing[0].ROV/self.routing[0].RAY_LEN_M
        non_zero_detected_ratios = detected_ratios[np.nonzero(detected_ratios)]
        ret = 0.0
        # --- Distance progress (potential-based shaping)
        progress_reward = prevd2destin - d2destin
        ret += progress_reward

        # --- Time penalty (encourages efficiency)
        ret += -0.01

        # --- Safe-distance penalty (soft constraint)
        if len(non_zero_detected_ratios) > 0 and any(non_zero_detected_ratios < safe_detected_ratio):
            # FIX: penalty is now ADDITIVE to progress, not replacing it
            safety_penalty = -1*((safe_detected_ratio - min(non_zero_detected_ratios)) / safe_detected_ratio)**2
            ret += safety_penalty

        # --- Terminal conditions (override)
        if d2destin < reachThreshold_m:
            ret = 10.0
        elif int(self.CONTACT_FLAGS[0]) == 1:
            ret = -10.0

        self.CUM_REWARD += ret
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        # cond2 : reached destination area
        # cond2 = np.linalg.norm(self.routing.DESTINATION.reshape(3,1) - state[0:3].reshape(3,1)) <= 0.5
        
        reachThreshold_m = 1

        if np.linalg.norm(self.routing[0].DESTINATION-state[0:3]) <= reachThreshold_m:
            return True
        elif int(self.CONTACT_FLAGS[0]) == 1:
            return True
        # FIX: timeout moved to _computeTruncated (timeout != failure for DRL)
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        # FIX: timeout is now truncation, not termination
        # This matters because DRL should NOT zero out value estimates on timeout
        if self.step_counter/self.PYB_FREQ >= self.EPISODE_LEN_SEC:
            return True

        # Out of bounds safety net
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 200 or abs(state[1]) > 200 or state[2] > 50:
            return True

        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict with rich metrics.

        Returns
        -------
        dict
            Keys: collision (bool), intrusion (bool), reached (bool),
            d2dest (float), min_obstacle_ratio (float).
        """
        state = self._getDroneStateVector(0)
        d2dest = float(np.linalg.norm(self.routing[0].DESTINATION - state[0:3]))
        collision = int(self.CONTACT_FLAGS[0]) == 1
        reached = d2dest < 1.0

        # Intrusion: any ray hit within ROV (range of vision) without collision
        detected_ratios = self.routing[0].RAYS_INFO[:, 0]
        safe_ratio = self.routing[0].ROV / self.routing[0].RAY_LEN_M
        nonzero = detected_ratios[np.nonzero(detected_ratios)]
        min_ratio = float(np.min(nonzero)) if len(nonzero) > 0 else 1.0
        intrusion = bool(min_ratio < safe_ratio) and not collision

        return {
            "collision": collision,
            "intrusion": intrusion,
            "reached": reached,
            "d2dest": d2dest,
            "min_obstacle_ratio": min_ratio,
        }
    
    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.
        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.
        """
        MAX_LIN_VEL_XY = self.SPEED_LIMIT
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z  # FIX: was MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    def _clipAndNormalizeRay(self,
                            rayinfo
                            ):
        """Normalizes a ray's informaiton to the [-1,1] range.
        Parameters
        ----------
        rayinfo : ndarray
            (5,)-shaped array of floats containing the non-normalized information of a SINGLE ray.
        Returns
        -------
        ndarray
            (5,)-shaped array of floats containing the normalized information of a SINGLE ray
        """
        num_info_extract = len(rayinfo)   # 3 or 5 [hit_ids, hit_fraction, hit_pos_x, hit_pos_y, hit_pos_z] per ray
        MAX_HIT_IDS = self.NUM_DRONES
        MIN_HIT_IDS = -1

        h2destin = np.linalg.norm(self.routing[0].DESTINATION - self.routing[0].HOME_POS)

        MAX_XY = h2destin
        MAX_Z = h2destin

        if num_info_extract == 5:
            clipped_hit_ids = np.clip(rayinfo[0], MIN_HIT_IDS, MAX_HIT_IDS)
            clipped_hit_fraction = rayinfo[1]  # no need (already in [0,1] range)
            clipped_hit_pos_xy = np.clip(rayinfo[2:4], -MAX_XY, MAX_XY)
            clipped_hit_pos_z = np.clip(rayinfo[4], 0, MAX_Z)

            normalized_hit_ids = clipped_hit_ids / MAX_HIT_IDS     # [-1, 1]
            normalized_hit_fraction = clipped_hit_fraction # [0, 1]
            normalized_hit_pos_xy = clipped_hit_pos_xy / MAX_XY  #[-1, 1]
            normalized_hit_pos_z = clipped_hit_pos_z / MAX_Z   #[0, 1]

            norm_and_clipped = np.hstack([normalized_hit_ids,
                                      normalized_hit_fraction,
                                      normalized_hit_pos_xy,
                                      normalized_hit_pos_z,
                                      ]).reshape(5,)
        elif num_info_extract == 3:
            # (hit_fraction, x, y)
            clipped_hit_fraction = rayinfo[0]  # no need (already in [0,1] range)
            clipped_hit_pos_xy = np.clip(rayinfo[1:3], -MAX_XY, MAX_XY)
            # clipped_hit_pos_z = np.clip(rayinfo[2], 0, MAX_Z)

            normalized_hit_fraction = clipped_hit_fraction # [0, 1]
            normalized_hit_pos_xy = clipped_hit_pos_xy / MAX_XY  #[-1, 1]
            # normalized_hit_pos_z = clipped_hit_pos_z / MAX_Z   #[0, 1]

            norm_and_clipped = np.hstack([
                                      normalized_hit_fraction,
                                      normalized_hit_pos_xy,
                                      ]).reshape(3,)
        return norm_and_clipped
    
    def _clipAndNormalizeD2Destin(self, d2destin, drone_id):
        h2destin = np.linalg.norm(self.routing[drone_id].DESTINATION - self.routing[drone_id].HOME_POS)
        MIN_D2DESTIN = 0
        MAX_D2DESTIN = h2destin
        clipped_d2destin = np.clip(d2destin, MIN_D2DESTIN, MAX_D2DESTIN)
        normalized_d2destin = clipped_d2destin / MAX_D2DESTIN  #range [0, 1]
        return normalized_d2destin
    
    ################################################################################

import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary

from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag, RouteStatus
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.PIDVelocityControl import PIDVelocityControl
from gym_pybullet_drones.routing.RouteMission import RouteMission

class ExtendedSARLAviary(RoutingAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 30,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 sensor_cfg: dict=None,
                 action_cfg: dict=None,
                 skip_drone_raycasting: bool=False,
                 obs_choice: str="sensor",
                 mission_cfg: dict=None
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        self.NUM_OTHER_DRONES = num_drones - 1
        # --- Store config dicts (with defaults) ---
        _ac = action_cfg or {}
        self._accel_value = _ac.get('accel_value', 2)
        self._decel_value = _ac.get('decel_value', -2)
        self._sensor_cfg = sensor_cfg
        self._mission_cfg = mission_cfg
        # =============================================================================
        homePos =  np.array([0,0,0.5]) 
        destin  =  np.array([0.2, 10, 1])
        self.HOME_POS = homePos
        self.DESTIN = destin
        num_drones_total = num_drones 

        self.MISSION = RouteMission()
        self.MISSION.generateRandomMission(maxNumDrone=num_drones, minNumDrone=num_drones,
                                           mission_cfg=self._mission_cfg)
        self.OBS_CHOICE = obs_choice  # ray, sensor, sector
        
        # =============================================================================

        #### Create a buffer for the last .5 sec of actions ########
        # self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        # self.ACTION_BUFFER_SIZE = 0
        # self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.COMPUTE_DONE = False
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.AUTOROUTING]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.HB]:
                # self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones_total)]
                self.ctrl = [PIDVelocityControl(drone_model=drone_model) for i in range(num_drones)]
                self.routing = [IFDSRoute(drone_model=drone_model, drone_id=i, sensor_cfg=self._sensor_cfg) for i in range(num_drones)]
                
                self.INIT_XYZS = self.MISSION.INIT_XYZS
                self.INIT_RPYS = self.MISSION.INIT_RPYS
                for j in range(num_drones):
                    self.INIT_XYZS[j,:] = self.MISSION.INIT_XYZS[j,:]
                    self.routing[j].HOME_POS = self.MISSION.INIT_XYZS[j,:]
                    self.routing[j].DESTINATION = self.MISSION.DESTINS[j,:]
                    self.routing[j].CUR_POS = self.MISSION.INIT_XYZS[j, :]
                    self.routing[j].CUR_RPY = self.MISSION.INIT_RPYS[j,:]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")

        #### Create a buffer for the last .5 sec of Sensors ########
        self.SENSOR_BUFFER_SIZE = 1   # 5: five informations from raycast (obj_id, hit_fraction, (hit_xyz))
        self.sensor_buffer = deque(maxlen=self.SENSOR_BUFFER_SIZE)
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones_total,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=self.MISSION.INIT_XYZS,
                         initial_rpys=self.INIT_RPYS,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         skip_drone_raycasting=skip_drone_raycasting,
                         )
        # Propagate GUI flag to routing objects so they can skip debug drawing
        for r in self.routing:
            r._gui = self.GUI
        #### Set a limit on the maximum target speed ###############

    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE == ActionType.AUTOROUTING:
            return spaces.Discrete(3, start = 0) # 3 discrete actions, details in _preprocessAction()
            # return spaces.Discrete(2, start = 0) # 2 discrete action
        
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        # p.removeAllUserDebugItems()
        # self.action_buffer.append(np.array([[float(action)]])) # Need to revise this to have N-number of drones
                                                        # (similar to [[discrete_act_lo] for i in range(self.NUM_DRONES)])])
                                                        
        rpm = np.zeros((self.NUM_DRONES, 4))
        
        for k in range(self.NUM_DRONES):  # k: num drone
            # Process action based on ACT_TYPE
            state = self._getDroneStateVector(k)

            # --- Set commands BEFORE guidance (FIX: was after, causing 1-step delay) ---
            if k == 0:
                # Agent drone: apply RL action
                if action == 0:
                    self.routing[0]._setCommand(RouteCommandFlag, "follow_global", 1)
                    self.routing[0]._setCommand(SpeedCommandFlag, "accelerate", self._accel_value)
                elif action == 1:
                    self.routing[0]._setCommand(RouteCommandFlag, "follow_global", 1)
                    self.routing[0]._setCommand(SpeedCommandFlag, "accelerate", self._decel_value)
                elif action == 2:
                    self.routing[0]._setCommand(RouteCommandFlag, "follow_global", 1)
                    self.routing[0]._setCommand(SpeedCommandFlag, "constant")
                else:
                    raise ValueError(f"Invalid action: {action}")
            else:
                # Non-agent drones: constant traffic on global route
                self.routing[k]._setCommand(RouteCommandFlag, "follow_global", 1)
                self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", 1)

            # --- Compute IFDS route (once at start, then follow global path) ---
            need_ifds = (self.routing[k].GLOBAL_PATH.size == 0)
            if need_ifds:
                foundPath, path = self.routing[k].computeRouteFromState(
                    route_timestep=self.routing[k].route_counter,
                    state=state,
                    home_pos=self.routing[k].HOME_POS,
                    target_pos=self.MISSION.DESTINS[k, :],
                    speed_limit=self.SPEED_LIMIT,
                    obstacle_data=self.OBSTACLE_DATA,
                    drone_ids=self.DRONE_IDS,
                )

                if self.routing[k].route_counter == 1:
                    if foundPath > 0:
                        self.routing[k].setGlobalRoute(path)
                    else:
                        fromPos = self.routing[k].HOME_POS
                        toPos = self.routing[k].DESTINATION
                        n_wp = 100
                        gpath = self.routing[k]._generateWaypoints(fromPos, toPos, n_wp)
                        self.routing[k].setGlobalRoute(np.array(gpath).reshape((3, n_wp)))
            else:
                # Reuse existing global path for all drones
                self.routing[k].setCurrentRoute(self.routing[k].GLOBAL_PATH)
                self.routing[k].route_counter += 1

            # --- Compute guidance (now processes the CURRENT action's commands) ---
            self.routing[k].computeGuidanceFromState(
                state=state,
                drone_ids=k,
                route_timestep=self.routing[k].route_counter,
                speed_limit=self.SPEED_LIMIT,
            )

            # --- PID velocity control ---
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_vel=self.routing[k].TARGET_VEL,
            )
            rpm[k, :] = rpm_k

        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.
        """
        # Base observation (7 vars) X Y Z Yaw VX VY VZ
        lo, hi = -1.0, 1.0
        obs_lower_bound = np.array([lo, lo, 0, lo, lo, lo, lo], dtype=float)
        obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi], dtype=float)
        # ++++++ Add distance-to-destination to observation space ++++++
        # Add distance-to-destination
        obs_lower_bound = np.append(obs_lower_bound, 0.0)
        obs_upper_bound = np.append(obs_upper_bound, np.inf)
        # obs_lower_bound = np.hstack([obs_lower_bound, np.array([[discrete_act_lo] for i in range(1)])])
        # obs_upper_bound = np.hstack([obs_upper_bound, np.array([[discrete_act_hi] for i in range(1)])])
        if self.OBS_CHOICE  == "ray":
            num_rays = self.routing[0].NUM_RAYS
            #++++++ Add ray reading to observation space +++++++++++++
            # Ray info: [obj_id, hit_fraction, hitPos_x, hitPos_y, hitPos_z] per ray
            # Extract only 3 per ray (hit_fraction, hitx, hity)
            sensing_lo = np.tile([0, -np.inf, -np.inf], num_rays)
            sensing_hi = np.tile([1, np.inf, np.inf], num_rays)
        elif self.OBS_CHOICE == "sensor":
            num_sensors = self.routing[0].NUM_SENSORS
            # Extracted Features: [r_min, r_mean, dhit, LOS normalized] per sensor
            sensing_lo = np.tile([0, 0, 0, -1], num_sensors)
            sensing_hi = np.tile([1, 1, 1, 1], num_sensors)
        elif self.OBS_CHOICE == "sector":
            num_sectors = self.routing[0].NUM_SECTORS
            # Extracted Features: [r_min, r_mean, dhit, los_angle] per sector
            sensing_lo = np.tile([0, 0, 0, -1], num_sectors)
            sensing_hi = np.tile([1, 1, 1, 1], num_sectors)
        else:
            print("[ERROR] in BaseRLAviary._observationSpace():  Invalid OBS_CHOICE")
        
        obs_lower_bound = np.concatenate([obs_lower_bound, sensing_lo])
        obs_upper_bound = np.concatenate([obs_upper_bound, sensing_hi])
        ############################################################
        obs_lower_bound =  obs_lower_bound.reshape(obs_lower_bound.shape[0],)
        obs_upper_bound =  obs_upper_bound.reshape(obs_upper_bound.shape[0],)
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        size_obs = self.observation_space.shape[0]

        # Get raw state and normalize kinematic features
        obs = self._getDroneStateVector(0)
        norm_state = self._clipAndNormalizeState(obs)  # FIX: was using raw unnormalized state
        self.routing[0]._batchRayCast(self.routing[0].DRONE_ID)
        d2destin = self.routing[0].getDistanceToDestin()
        d2destin_normalized = self._clipAndNormalizeD2Destin(d2destin, drone_id=0)

        if self.OBS_CHOICE == "ray":
            sensing_matrix = self.routing[0].RAYS_INFO
            sensing_normalized = np.apply_along_axis(self._clipAndNormalizeRay, 1, sensing_matrix).reshape(-1)
        elif self.OBS_CHOICE == "sensor":
            sensing_normalized = self.routing[0].SENSOR_INFO.reshape(-1)
        elif self.OBS_CHOICE == "sector":
            sensing_normalized = self.routing[0].SECTOR_INFO.reshape(-1)
        else:
            raise ValueError(f"[Error] in ExtendedSARLAviary - Invalid OBS_CHOICE")

        # Build 1D observation vector using NORMALIZED kinematic states
        # 7 kinematic features: X, Y, Z, Yaw, Vx, Vy, Vz
        obs_flat = np.hstack([
            norm_state[0:3],      # X, Y, Z
            norm_state[9:10],     # Yaw
            norm_state[10:13],    # Vx, Vy, Vz
            d2destin_normalized,
            sensing_normalized,
        ]).astype(np.float32)

        # FIX: return shape (N,) not (1, N) for standard Gym compatibility
        assert obs_flat.shape == (size_obs,), f"Obs shape mismatch: {obs_flat.shape} vs ({size_obs},)"
        return obs_flat

    ################################################################################

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        self.CUM_REWARD = 0
        
        self.MISSION.generateRandomMission(maxNumDrone=self.NUM_DRONES, minNumDrone=self.NUM_DRONES,
                                           seed=seed, mission_cfg=self._mission_cfg)
        
        p.resetSimulation(physicsClientId=self.CLIENT)
        
        #### Housekeeping ##########################################
        self._housekeeping()
        self.step_counter = 0
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        for j in range(self.NUM_DRONES):
            self.routing[j].reset()
            self.ctrl[j].reset()
            self.routing[j].CUR_POS = self.MISSION.INIT_XYZS[j,:]
            self.routing[j].CUR_VEL = np.array([0,0,0])
            self.routing[j].CUR_RPY = self.MISSION.INIT_RPYS[j,:]
            self.routing[j].HOME_POS = self.MISSION.INIT_XYZS[j,:]
            # self.routing[j].GLOBAL_PATH = np.array([])
            self.routing[j].DESTINATION = self.MISSION.DESTINS[j,:]
            self.CONTACT_FLAGS[j] = 0
            self.INIT_XYZS[j,:] = self.MISSION.INIT_XYZS[j,:]
            self.INIT_RPYS[j,:] = self.MISSION.INIT_RPYS[j,:]

        self.OBSTACLE_DATA = {}
        self._getObstaclesData()
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        self._detectCollision()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a state to the [-1,1] range.
        Must be implemented in a subclass.
        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized information of a single ray.
        """
        raise NotImplementedError
    
    def _clipAndNormalizeD2Destin(self, d2destin, drone_id):
        """Normalize a dinstance to destination to the [0, 1] range"""
        raise NotImplementedError
    
    def _clipAndNormalizeRay(self,rayinfo):
        """Normalizes a drone's state to the [-1,1] range.
        Must be implemented in a subclass.
        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.
        """
        raise NotImplementedError
    
    def _clipAndNormalizeSensor(self, sensorinfo):
        """Normalizes a drone's state to the [-1,1] range.
        Must be implemented in a subclass.
        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.
        """
        raise NotImplementedError

    def _clipAndNormalizeSector(self, sectorinfo):
        """Normalizes a drone's state to the [-1,1] range.
        Must be implemented in a subclass.
        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.
        """
        raise NotImplementedError
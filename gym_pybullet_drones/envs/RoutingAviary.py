import os
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from PIL import Image
from gym_pybullet_drones.envs.SceneCreator import SceneCreator

class RoutingAviary(BaseAviary):
    """Multi-drone environment class for control applications."""
    
    OBSTACLE_IDS = set([])

    ################################################################################

    def __init__(self,
                  drone_model: DroneModel=DroneModel.CF2X,
                  num_drones: int=1,
                  neighbourhood_radius: float=np.inf,
                  initial_xyzs=None,
                  initial_rpys=None,
                  physics: Physics=Physics.PYB,
                  pyb_freq: int = 240,
                  ctrl_freq: int = 240,
                  gui=False,
                  record=False,
                  obstacles=True,
                  user_debug_gui=True,
                  vision_attributes=False,
                  output_folder='results',
                  skip_drone_raycasting: bool=False
                  ):
        """Initialization of a generic aviary environment.

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
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.

        """
        super().__init__(drone_model=drone_model,
                          num_drones=num_drones,
                          neighbourhood_radius=neighbourhood_radius,
                          initial_xyzs=initial_xyzs,
                          initial_rpys=initial_rpys,
                          physics=physics,
                          pyb_freq=pyb_freq,
                          ctrl_freq=ctrl_freq,
                          gui=gui,
                          record=record,
                          obstacles=obstacles,
                          user_debug_gui=user_debug_gui,
                          output_folder=output_folder
                          )
        
        #### Set a limit on the maximum target speed ###############
        speedLimitingFactor = 1  #0.1 # 0.03
        self.SPEED_LIMIT = speedLimitingFactor * self.MAX_SPEED_KMH * (1000/3600)
        self.CONTACT_POINTS = [() for _ in range(self.NUM_DRONES)]
        self.CONTACT_FLAGS = np.zeros(self.NUM_DRONES, dtype=int)
        
        self.SKIP_DRONE_RAYCASTING = skip_drone_raycasting
        self.OBSTACLE_DATA = {}
        self._getObstaclesData()
        
    ################################################################################
    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.performCollisionDetection(physicsClientId=self.CLIENT)
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            #*** Added performCollisionDetection ***
            if self.PHYSICS != Physics.DYN:
                # self._applyForceToObstacle()  # Apply force to obstacle (dynamic obstacles) 
                p.performCollisionDetection(physicsClientId=self.CLIENT)
                p.stepSimulation(physicsClientId=self.CLIENT)
                
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._applyForceToObstacle()
        self._updateAndStoreKinematicInformation()
        self._detectCollision()
        
  
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self._getObstaclesData()
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        # print(f"Stepping . . . step_counter = {self.step_counter}")
        return obs, reward, terminated, truncated, info
    
    ################################################################################
    def _addObstacles(self):
        """Add obstacles to the environment.
        These obstacles are loaded from standard URDF files included in Bullet.
        """
        scene = 0  # 3 for final testing
        return SceneCreator(self, scene).create()
    
    ################################################################################
    def _detectCollision(self):
        # print("Processing Collision Detection . . .")
        for i in range(self.NUM_DRONES):
            self.CONTACT_POINTS[i] = p.getContactPoints(self.DRONE_IDS[i])
  
            if len(self.CONTACT_POINTS[i]) > 0:
                self.CONTACT_FLAGS[i] = 1
                # print("Agent" + str(i) + ": Collided !!!!!!!!!!!!!!!")
            else:
                self.CONTACT_FLAGS[i] = 0

    ################################################################################   
    def _getObstaclesData(self):
        obstacleList = list(RoutingAviary.OBSTACLE_IDS)
        # Store obstacles data
        self._storeObjectData(obstacleList, scale=1)
        # Always include agent drone (index 0) in raycasting
        # Skip non-agent drones only when skip_drone_raycasting is enabled
        droneList = [self.DRONE_IDS[0]]  # Always include agent drone
        if not self.SKIP_DRONE_RAYCASTING:
            droneList.extend(self.DRONE_IDS[1:])  # Include all drones when not optimizing
        self._storeObjectData(droneList, scale=1/10)
        
    def _storeObjectData(self, objectList, scale=1):
        for j in range(len(objectList)):
            pos, orn = p.getBasePositionAndOrientation(objectList[j])
            visualShapeData = p.getVisualShapeData(objectList[j])
            objectSize = tuple(np.array(visualShapeData[0][3])*scale)
            self.OBSTACLE_DATA[str(objectList[j])] = {"position": pos,
                                                      "size": objectSize}
    ################################################################################
    def _applyForceToObstacle(self):
        # Apply force to object -> dynamic obstacles
        t = self.step_counter/self.PYB_FREQ
        altering_sec = 2  #1.5
        sign = -1 if (t // altering_sec) % 2 == 0 else 1
        if len(self.OBSTACLE_IDS) != 0:
            for id in self.OBSTACLE_IDS:
                if id%2 == 0:
                    sign *= -1
                obj_info = p.getDynamicsInfo(id, -1)
                mass = obj_info[0]
                pos, orn = p.getBasePositionAndOrientation(id)
                p.applyExternalForce(id, -1, [20*sign,0,-mass*9.81], pos, flags=p.WORLD_FRAME) 
        # else:
            # print("[ERROR] in RoutingAviary, No obstacles")  
    
    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        raise NotImplementedError
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        raise NotImplementedError

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        # adjacency_mat = self._getAdjacencyMatrix()
        # return {str(i): {"state": self._getDroneStateVector(i), "neighbors": adjacency_mat[i, :]} for i in range(self.NUM_DRONES)}
        raise NotImplementedError

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        # clipped_action = np.zeros((self.NUM_DRONES, 4))
        # for k, v in action.items():
        #     clipped_action[int(k), :] = np.clip(np.array(v), 0, self.MAX_RPM)
        # return clipped_action
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        raise NotImplementedError
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years    
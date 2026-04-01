import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources
from enum import Enum
import pybullet as p


from gym_pybullet_drones.utils.enums import DroneModel

class RouteStatus(Enum):
    GLOBAL = "global route"
    LOCAL  = "local route"                 
class SpeedStatus(Enum):
    CONSTANT   = "constant speed"
    ACCELERATE = "accelerating" 
    DECELERATE = "decelerating"
    HOVERING   = "hovering"
class RouteCommandFlag(Enum):
    CHANGE = "change_route"
    FOLLOW_GLOBAL = "follow_global"
    FOLLOW_LOCAL  = "follow_local"
    NONE = "none"
class SpeedCommandFlag(Enum):
    ACCEL         = "accelerate"
    CONST         = "constant"
    HOVER         = "hover"
    NONE          = "none"

class CommandTypeError(KeyError): pass
class CommandValueError(ValueError): pass
    
class Commander:
    classList = [SpeedStatus, SpeedCommandFlag, RouteCommandFlag]
    
    def __init__(self, commandType, command_str, value=None):
        if commandType not in Commander.classList:
            raise CommandTypeError("Invalid command type")

        command = self._parse_command(command_str, commandType)
        
        self._value = value
        self._name = command.value
        self._type = commandType

    def _parse_command(self, command_str, commandType):
        for member in commandType:
            if member.value == command_str:
                return member
        raise CommandValueError(f"Invalid command '{command_str}' for the command type {commandType}")


class BaseRouting(object):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeRouteFromState()`,
    the main method `computeRoute()` should be implemented by its subclasses.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 drone_id,
                 g: float=9.8,
                 sensor_cfg: dict=None
                 ):
        """Common Routing classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.
        sensor_cfg : dict, optional
            Sensor parameters from YAML config. Keys: num_sensors, num_rays_per_sensor,
            sensor_fov_deg, ray_len_m, rov. If None, uses hardcoded defaults.

        """
        #### Set general use constants #############################
        self.DRONE_MODEL = drone_model
        self.DRONE_ID = drone_id
        self._gui = False  # Set to True by environment when GUI is active
        """DroneModel: The type of drone to control."""
        self.GRAVITY = g*self._getURDFParameter('m')
        """float: The gravitational force (M*g) acting on each drone."""
        self.GLOBAL_PATH = np.array([])
        self.CURRENT_PATH = np.array([])
        """ndarray (3,N) : The global static route of UAV from starting to destination"""
        self.CUR_POS = np.array([0,0,0])
        self.CUR_VEL = np.array([0,0,0])
        self.CUR_RPY = np.array([0,0,0])
        self.DESTINATION = np.array([0,0,0])
        self.TARGET_POS   = np.array([])   # Check-> initialize with empty array should work
        self.TARGET_VEL  =  np.array([0,0,0])
        self.HOME_POS = np.array([0,0,0])
        self.STAT = [RouteStatus.GLOBAL, SpeedStatus.CONSTANT]
        
        self.COMMANDS = [Commander(RouteCommandFlag, "none"), Commander(SpeedCommandFlag, "none")]
        self._resetAllCommands()
        self.route_counter = 0
        self.DETECTED_OBS_IDS = []
        self.DETECTED_OBS_DATA = {}

        
        _sc = sensor_cfg or {}
        self.NUM_RAYS_PER_SENSOR = _sc.get('num_rays_per_sensor', 13)
        self.NUM_SENSORS = _sc.get('num_sensors', 10)
        self.NUM_SECTORS = 8
        self.SENSOR_FOV_DEG = _sc.get('sensor_fov_deg', 20)
        self.NUM_RAYS = self.NUM_SENSORS * self.NUM_RAYS_PER_SENSOR
        self.RAY_LEN_M = _sc.get('ray_len_m', 30)
        self.ROV = _sc.get('rov', 12.77)
        self.SFG = 3

        # Tracks consecutive static actions (used in reward function)
        self.static_action_counter = 0
        
        self.reset()

    ################################################################################

    def reset(self):
        """Reset the routing classes.

        A general use counter is set to zero.

        """
        self.static_action_counter = 0
        self.GLOBAL_PATH = np.array([])
        self.CURRENT_PATH = np.array([])
        self.route_counter = 0

    ################################################################################

    def computeRouteFromState(self,
                            route_timestep,
                            state,
                            home_pos,
                            target_pos,
                            speed_limit,
                            obstacle_data=None,
                            drone_ids = np.array([1])
                            ):
        """Interface method using `computeRoute`.

        It can be used to compute a route directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        route_timestep : float
            The time step at which the route is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        obstacles_pos : ndarray
            (N,3)-shaped array of floats containing obstacles' positions. The first one is environment
        obstacles_size : ndarray
            (N,3)-shaped array of floats containing obstacles' sizes. The first one is environment
        """
        self.HOME_POS= home_pos
        self.DESTINATION = target_pos
        
        self._processDetection(obstacle_data)
        
        return self.computeRoute(route_timestep=route_timestep,
                                cur_pos=state[0:3],
                                target_pos = target_pos,
                                obstacle_data = self.DETECTED_OBS_DATA,
                                )

    ################################################################################

    def computeRoute(self,
                     route_timestep,
                     cur_pos,
                     target_pos,
                     obstacle_data,
                     ):
        """Abstract method to compute the route for a single drone.

        It must be implemented by each subclass of `BaseRoute`.

        Parameters
        ----------
        route_timestep : float
            The time step at which the route is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.

        """
        raise NotImplementedError

################################################################################
################################################################################
    
    def getDistanceToDestin(self):
        cur_pos = self.CUR_POS.reshape(1,3)
        destin = self.DESTINATION.reshape(1,3)
        return np.linalg.norm(cur_pos - destin)

    ################################################################################

    def _plotRoute(self, path):
        """Plot the current route using PyBullet's debug lines.

        Parameters
        ----------
        path : ndarray
            (3, N) array of 3D waypoints.
        """
        # Skip plotting if GUI is not active
       
        if not self._gui:
            return
            
        if self.DRONE_ID == 0:
            pathColor = [0.1, 0.1, 1]
        else:
            pathColor = [0.5, 0.5, 0.6]

        stepper = 1
        for i in range(0, path.shape[1] - stepper, stepper):
            p.addUserDebugLine(path[:, i], path[:, i + stepper], pathColor, lineWidth=5, lifeTime=0.05)


    def setIFDSCoefficients(self, rho0_ifds=None, sigma0_ifds=None, sf_ifds=None):
        """Sets the coefficients of the IFDS path planning algorithm.

        This method throws an error message and exist is the coefficients
        were not initialized (e.g. when the routing algorithm is not the IFDS).

        Parameters
        ----------
        rho0_ifds : float, optional
            Normal repulsive coefficients (minimum is 0.1).
        sigma0_ifds : float, optional
            Tangential repulsive coefficients (minimum is 0.1).
        sf : boolean, optional
            Shape-following feature activation.
        """
        ATTR_LIST = ['RHO0_IFDS', 'SIGMA0_IFDS', 'SF_IFDS']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] in BaseRouting.setIFDSCoefficients(), not all IFDS coefficients exist as attributes in the instantiated routing class.")
            exit()
        else:
            self.RHO0_IFDS = self.RHO0_IFDS if rho0_ifds is None else rho0_ifds
            self.SIGMA0_IFDS = self.SIGMA0_IFDS if sigma0_ifds is None else sigma0_ifds
            self.SF_IFDS = self.SF0_IFDS if sf_ifds is None else sf_ifds

    ################################################################################
    
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+URDF)
        URDF_TREE = etxml.parse(path).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
        
    ################################################################################
        
    def _setCommand(self, commandType, command_str, value=None):
        def _parse_command(command_str, commandType):
            for member in commandType:
                if member.value == command_str:
                    return member
            raise CommandValueError(f"Invalid command '{command_str}' for the command type {commandType}")
            
        classList = [RouteCommandFlag, SpeedCommandFlag]
        
        if commandType not in classList:
            raise CommandTypeError("Invalid command type")
        elif commandType == RouteCommandFlag:
            idx = 0
        elif commandType == SpeedCommandFlag:
            idx = 1
        
        # Update the COMMANDS attribute    
        self.COMMANDS[idx] = Commander(commandType, command_str, value)
        self._processCommand()
        
    ################################################################################
    
    def _processCommand(self):
        """Process the command and reset"""
        
        # First command position: route command
        if self.COMMANDS[0]._name != 'none':
            self._processRouteCommand()
            
        if self.COMMANDS[1]._name != 'none':
            self._processSpeedCommand()
        # else:
        #     print("No SpeedCommand")
                
    
    def _processRouteCommand(self):
        if self.COMMANDS[0]._name == RouteCommandFlag.CHANGE.value:
            self.switchRoute()
        elif self.COMMANDS[0]._name == RouteCommandFlag.FOLLOW_GLOBAL.value:
            self.STAT[0] = RouteStatus.GLOBAL
            self.SIM_MODE = 2
            self.PATH_OPTION = self.COMMANDS[0]._value
            self.setCurrentRoute(self.GLOBAL_PATH)
        elif self.COMMANDS[0]._name == RouteCommandFlag.FOLLOW_LOCAL.value:
            self.STAT[0] = RouteStatus.LOCAL
            self.SIM_MODE = 1
            self.PATH_OPTION = self.COMMANDS[0]._value
        else:
            print("[Error] in _processRouteCommand()")
            
        # Reset the route command
        # self._resetRouteCommand()
            
    def _processSpeedCommand(self):
        if self.COMMANDS[1]._name == SpeedCommandFlag.ACCEL.value:
            # accelerate
            if self.COMMANDS[1]._value > 0:
                # print("Accelerating . . .")
                self.STAT[1] = SpeedStatus.ACCELERATE
            elif self.COMMANDS[1]._value < 0:
                # print("Decelerating . . .")
                self.STAT[1] = SpeedStatus.DECELERATE
            elif self.COMMANDS[1]._value == 0:
                # print("Constant Speed . . .")
                self.STAT[1] = SpeedStatus.CONSTANT
                self.TARGET_VEL = np.zeros(3)
            
        elif self.COMMANDS[1]._name == SpeedCommandFlag.CONST.value:
            # print("Constant Speed . . .")
            # cur_vel_unit = self.CUR_VEL /  np.linalg.norm(self.CUR_VEL)
            # self.TARGET_VEL = cur_vel_unit *self.COMMANDS[1]._value
            # self.TARGET_VEL = np.zeros(3)
            self.STAT[1] = SpeedStatus.CONSTANT
            
        elif self.COMMANDS[1]._name == SpeedCommandFlag.HOVER.value:
            # hover
            # FIX: Always re-latch HOVER_POS when entering hover mode.
            # Previously, the first-entry guard caused HOVER_POS to be set
            # from stale CUR_POS (before computeGuidanceFromState updates it).
            # Now we re-latch every time we transition into hover, so the last
            # _processCommand call (with fresh CUR_POS) wins.
            if self.STAT[1] != SpeedStatus.HOVERING:
                self.HOVER_POS = self.CUR_POS.copy()
                self.STAT[1] = SpeedStatus.HOVERING
            self.TARGET_POS = self.HOVER_POS
            self.TARGET_VEL = np.zeros(3)
        else:
            print("[Error] in _processSpeedCommand()")
        
        # Reset the speed command (NEED)
        # self._resetSpeedCommand() 
                
    def switchRoute(self):
        """Switch current route from global to local, or from local to global"""
        if self.STAT[0].value == RouteStatus.GLOBAL.value:
            # print("Switching to Local route")
            # self.STAT[0] = RouteStatus.LOCAL
            self.SIM_MODE = 1
            
            self.COMMANDS[0]._name = RouteCommandFlag.FOLLOW_LOCAL.value
            self._processRouteCommand()
        
        elif self.STAT[0].value == RouteStatus.LOCAL.value:
            # print("Switching to Global route")
            # self.STAT[0] = RouteStatus.GLOBAL
            self.SIM_MODE =2
            
            self.COMMANDS[0]._name = RouteCommandFlag.FOLLOW_GLOBAL.value
            self._processRouteCommand()
             
        else:
            print("[Error] in switchRoute()")
        
    def _resetAllCommands(self):
        self._resetRouteCommand()
        self._resetSpeedCommand()
        
    def _resetRouteCommand(self):
        self.COMMANDS[0] = Commander(RouteCommandFlag, "none")
    
    def _resetSpeedCommand(self):
        self.COMMANDS[1] = Commander(SpeedCommandFlag, "none")

    ################################################################################
    def setGlobalRoute(self, route):
        """Store global route
        Parmaters
        ---------
        route : ndarray
            (3,N)-shaped array of floats containing the global route
        """
        self.GLOBAL_PATH = route
        # print("Setting a global route")

    def setCurrentRoute(self, route):
        """Store current route
        Parmaters
        ---------
        route : ndarray
            (3,N)-shaped array of floats containing the current route
        """
        self.CURRENT_PATH = route
    
    ################################################################################
    
    def _updateCurPos(self, pos):
        self.CUR_POS = pos
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=pos)
    def _updateCurRpy(self, rpy):
        self.CUR_RPY = rpy    
    def _updateCurVel(self, vel):
        self.CUR_VEL = vel
        
    def _batchRayCast(self, drone_ids):
        """
        Update self.DETECTED_OBS_IDS based on batch ray casting. DETECTED_OBS_IDS is a list of 
        detected obstacle's id

        Returns:
            None.

        """
        # rayHitColor = [0, 1, 0]
        rayHitColor = [0,1,0]
        # rayMissColor = [1, 1, 0.1]  # yellow
        rayMissColor = [0.5, 0.5, 1]  # green
        replaceLines = False
        # rayFrom = self.CUR_POS
        # p.removeAllUserDebugItems()

        detected_obs_ids = []
        rayTo = []
        rayIds = []
        rayFrom = [self.CUR_POS for _ in range(self.NUM_RAYS)]
        
        

        # -- number of rays : check from len(results)
        # -- results is a tuple of tuples
        # ******** Select the Lidar shape *****************
        # rayTo = self._RayCast_Sphere(rayFrom)
        # rayTo = self._RayCast_Circle(rayFrom)
        rayTo = self._RayCast_Circle_FoV(rayFrom)
        results = p.rayTestBatch(rayFrom, rayTo, numThreads=1)
        # *************************************************
        
        self.RAYS_INFO = self._extractRayInfo(results)
        self.SENSOR_INFO = self._extractSensorInfo(results, rays_per_sensor=self.NUM_RAYS_PER_SENSOR)
        self.SECTOR_INFO = self._extractSectorInfo(results, n_sectors=self.NUM_SECTORS, plot_edges=True)

        min_detect_ratio = self.ROV / self.RAY_LEN_M
        
        # if (not replaceLines):
        # p.removeAllUserDebugItems()
        for i in range(self.NUM_RAYS):
            hitObjectUid = results[i][0]
            
            if (hitObjectUid < 0):
                hitPosition = [float('inf'), float('inf'), float('inf')]
                if self.DRONE_ID == 0 and self._gui:
                    p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, lifeTime=0.02, lineWidth=2)
            else:
                # Whether detection of other fellow UAVs is allowed or not
                detectOtherUAV = True
                if detectOtherUAV:
                    # hitCondition = hitObjectUid!=0 # ignore the floor
                    hitCondition = hitObjectUid!=0  and hitObjectUid != self.DRONE_ID+1
                    # hitCondition = hitObjectUid != self.DRONE_ID+1
                else:
                    # hitCondition = hitObjectUid!=0 and hitObjectUid not in drone_ids
                    hitCondition = hitObjectUid not in drone_ids

                if hitCondition:
                # if hitObjectUid!=0:
                    detected_obs_ids.append(hitObjectUid) if hitObjectUid not in detected_obs_ids and hitObjectUid != 0 else detected_obs_ids
                    hitPosition = results[i][3]
                    # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
                    # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, lineWidth=1)

                    # if self.DRONE_ID == 0:
                    #     if self.RAYS_INFO[i,1] < 0.2:
                    #         rayHitColor = [1, 0, 0]
                    #     p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, lineWidth=2, lifeTime=0.02)
                    obj_dist = np.linalg.norm(rayFrom[i] -self.RAYS_INFO[i,0:3])
                    if obj_dist < self.ROV:
                        # Plot red rays if drones intrude other's Operational Volume Radius
                        rayHitColor = [1, 0, 0]
                    if self.DRONE_ID == 0 and self._gui:
                        p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, lineWidth=2, lifeTime=0.02)
    
        self.DETECTED_OBS_IDS = detected_obs_ids

    def _RayCast_Sphere(self, rayFrom):
        numRays = self.NUM_RAYS
        rayLen = self.RAY_LEN_M
        # sunflower on a sphere: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
        indices = np.arange(0, numRays, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices/numRays)
        theta = np.pi * (1 + 5**0.5) * indices

        x, y, z = rayLen* np.cos(theta) * np.sin(phi), rayLen* np.sin(theta) * np.sin(phi), rayLen*np.cos(phi)
        rayTo = [[self.CUR_POS[0]+x[i], self.CUR_POS[1]+y[i], self.CUR_POS[2]+z[i]] for i in range(numRays)]
        # rayIds = [p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor) for i in range(numRays)]
        
        return rayTo
    
    def _RayCast_Circle(self, rayFrom):
        numRays = self.NUM_RAYS
        rayLen = self.RAY_LEN_M
        rayTo = []

        blindspot_angle_deg = 0 # Angle of blind spot behind the vehicle (0 = no blind spot)
        ray_swept_angle_deg = 90-blindspot_angle_deg/2  
        start_angle = -(ray_swept_angle_deg) * np.pi/180  
        end_angle =  (180 + ray_swept_angle_deg) * np.pi/180 
        angle_range = end_angle - start_angle

        for i in range(numRays):
            angle = start_angle + angle_range * i / (numRays - 1)
            rayTo.append([
                self.CUR_POS[0] + rayLen * np.cos(angle),
                self.CUR_POS[1] + rayLen * np.sin(angle),
                self.CUR_POS[2]
            ])
        # rayIds = [p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor) for i in range(numRays)]
        return rayTo
    
    
    def _RayCast_Circle_FoV(self, rayFrom):
        numSensors = self.NUM_SENSORS
        raysPerSensor = self.NUM_RAYS_PER_SENSOR      # rays per sensor
        fov = self.SENSOR_FOV_DEG * np.pi/180  # FoV per sensor in radians
        rayLen = self.RAY_LEN_M
        rayTo = []
        rayAngleList = []
        cur_rpy = self.CUR_RPY 
        droneYaw = cur_rpy[2]  # assuming Z is yaw

        # Drone position
        sx, sy, sz = self.CUR_POS

        # Generate sensors
        for s in range(numSensors):
            # Sensor yaw (rotated evenly around circle, aligned with droneYaw)
            sensorAngle = 2 * np.pi * s / numSensors
            sensorYaw = sensorAngle + droneYaw

            # Rays within FoV (spread around sensor direction)
            for r in range(raysPerSensor):
                offset = -fov / 2 + fov * (r / (raysPerSensor - 1))
                rayAngle = sensorYaw + offset

                dx = rayLen * np.cos(rayAngle)
                dy = rayLen * np.sin(rayAngle)
                dz = 0
                toPoint = [sx + dx, sy + dy, sz + dz]

                rayAngleList.append(rayAngle)
                rayTo.append(toPoint)

        # Wrap all ray angles once at the end
        self.RAY_ANGLES = (np.array(rayAngleList) + np.pi) % (2 * np.pi) - np.pi

        return rayTo


    
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    ################################################################################
    def _extractRayInfo(self, rayResult):
        """
        Extract useful infomation from batch ray-casting to feed into agent observation

        Args:
            rayResult (tuple): tuple of tuples of raytest query returned from rayTestBatch
                The rayResult should have the dimension of 5 x (number of rays)
                which include (objectUniqueId, linkIndex, hit fraction, hit position, hit normal)

        Returns:
            array 
                Extracted information consisting of [hit_ids, hit_fraction, hit_pos_x, hit_pos_y, hit_pos_z] per ray
        """

        tempList = []
        for result in rayResult:
            hit_ids = result[0]  # -1 if no hit, positive integer if hit
            hit_fraction = result[2]  # range [0, 1] along the ray
            hit_pos = result[3]   # vec3, list of 3 floats (hit position in Cartesian world coordinate)
            # tempList.extend((hit_ids, hit_fraction, hit_pos[0], hit_pos[1], hit_pos[2]))
            tempList.extend(( hit_fraction, hit_pos[1], hit_pos[2] ))
            # tempList.extend(( hit_pos[0], hit_pos[1], hit_pos[2] ))
        
        # return np.array(tempList).reshape(self.NUM_RAYS, 5)
        return np.array(tempList).reshape(self.NUM_RAYS, 3)
    
    #########################################################################################################################
    def _extractSensorInfo(self, rayResult_original, rays_per_sensor):
        """
        Extract sensor-based features (rmin, rmean, hit_density, los_angle) from batch ray-casting.

        los_angle:
            Relative bearing (horizontal LOS angle) of the closest hit within the sensor
            w.r.t. drone yaw, normalized to [-1, 1] by dividing by pi.

        Each sensor consists of a fixed number of rays (e.g., 13). Rays are assumed
        to be ordered consecutively per sensor.

        Args:
            rayResult_original (tuple/list): batch ray-casting result from rayTestBatch
            rays_per_sensor (int): number of rays per sensor

        Returns:
            np.ndarray:
                Sensor-based features of shape (num_sensors * 4,)
                Each sensor contributes [rmin, rmean, hit_density, los_angle]
        """
        def _wrap_to_pi(angle: float) -> float:
            """Wrap angle to [-pi, pi]."""
            return (angle + np.pi) % (2.0 * np.pi) - np.pi
        max_range = self.RAY_LEN_M
        NUM_RAYS = self.NUM_RAYS
        assert NUM_RAYS % rays_per_sensor == 0, \
            "Number of rays must be divisible by rays_per_sensor"

        num_sensors = NUM_RAYS // rays_per_sensor
        yaw = float(self.CUR_RPY[2])

        # --- Drone position needed for LOS from hit point ---
        # Change this if your position variable name differs.
        drone_pos = np.asarray(self.CUR_POS, dtype=float)  # shape (3,)

        # Pre-allocate arrays
        ranges = np.zeros(NUM_RAYS, dtype=float)
        mask = np.zeros(NUM_RAYS, dtype=bool)

        # Store per-ray hit positions (only meaningful when mask[i] is True)
        hit_positions = np.zeros((NUM_RAYS, 3), dtype=float)

        # Extract per-ray info
        # Ray result original: (ObjectID, linkIndex, Hit fraction, (hitx, hity, hitz), (hitnormalx, hitnormaly, hitnormalz))
        for i, result in enumerate(rayResult_original):
            hit_id = result[0]
            hit_fraction = result[2]
            hit_pos = result[3]

            valid = (hit_id >= 0) and (hit_fraction > 0)
            mask[i] = valid

            # Range
            ranges[i] = hit_fraction * max_range

            # Hit position (only valid if hit occurred)
            if valid:
                hit_positions[i, :] = np.asarray(hit_pos, dtype=float)

        features = []

        # Aggregate per sensor
        for s in range(num_sensors):
            start = s * rays_per_sensor
            end = start + rays_per_sensor

            sensor_mask = mask[start:end]
            sensor_ranges = ranges[start:end]

            beams_in_sensor = rays_per_sensor
            cnt_hits = int(sensor_mask.sum())
            hit_density = cnt_hits / float(beams_in_sensor)

            if cnt_hits > 0:
                # rmin/rmean over valid returns
                r_valid = sensor_ranges[sensor_mask]
                rmin = float(r_valid.min() / max_range)
                rmean = float(r_valid.mean() / max_range)

                # LOS angle from the closest hit in this sensor
                hit_local_idx = np.where(sensor_mask)[0]          # indices 0..rays_per_sensor-1
                hit_ranges_local = sensor_ranges[sensor_mask]
                k_local = int(hit_local_idx[np.argmin(hit_ranges_local)])
                k_global = start + k_local

                hit_vec = hit_positions[k_global, :] - drone_pos  # vector from drone to hit
                bearing_global = float(np.arctan2(hit_vec[1], hit_vec[0]))
                bearing_rel = _wrap_to_pi(bearing_global - yaw)

                los_angle = bearing_rel / np.pi  # normalize to [-1, 1]
                
                #=============== For debugging ==============================
                # vx, vy = self.CUR_VEL[0], self.CUR_VEL[1]  # or wherever you store velocity
                # speed_xy = np.hypot(vx, vy)
                # if speed_xy > 1e-3:
                #     vel_yaw = np.arctan2(vy, vx)
                #     bearing_vel_rel = _wrap_to_pi(bearing_global - vel_yaw)
                #     los_vel = bearing_vel_rel / np.pi
                # else:
                #     vel_yaw = None
                #     los_vel = None

                # print(f"[sensor {s}] yaw={yaw:.3f} vel_yaw={(vel_yaw if vel_yaw is not None else np.nan):.3f} "
                #     f"bearing_global={bearing_global:.3f} los_yaw={los_angle:.3f} los_vel={(los_vel if los_vel is not None else np.nan):.3f} "
                #     f"v_xy=({vx:.2f},{vy:.2f})")
                #============================================================
            else:
                rmin = 1.0
                rmean = 1.0
                los_angle = 0.0  # neutral when no hit

            features.extend([rmin, rmean, hit_density, float(los_angle)])

        return np.array(features, dtype=float)
    #########################################################################################################################
    def _extractSectorInfo(self, rayResult, n_sectors=8, plot_edges=True):
        """
        Extract sector-based features from batch ray-casting.
        Per sector: [min_range, mean_range, hit_fraction, los_angle]
        los_angle is the bearing to the closest hit in the sector,
        normalized to [-1, 1] relative to drone yaw (see _extractSensorInfo).
        """
        def _wrap_to_pi(angle: float) -> float:
            """Wrap angle to [-pi, pi]."""
            return (angle + np.pi) % (2.0 * np.pi) - np.pi

        max_range = self.RAY_LEN_M
        yaw = float(self.CUR_RPY[2])
        drone_pos = np.asarray(self.CUR_POS, dtype=float)

        NUM_RAYS = self.NUM_RAYS

        # Initialize arrays
        ranges = np.zeros(NUM_RAYS, dtype=float)
        angles = np.zeros(NUM_RAYS, dtype=float)
        mask = np.zeros(NUM_RAYS, dtype=bool)
        hit_positions = np.zeros((NUM_RAYS, 3), dtype=float)

        # Compute ranges and angles relative to drone
        for i, result in enumerate(rayResult):
            hit_id = result[0]
            hit_fraction = result[2]
            hit_pos = result[3]

            valid = (hit_id >= 0) and (hit_fraction > 0)
            mask[i] = valid
            ranges[i] = hit_fraction * max_range
            angles[i] = self.RAY_ANGLES[i]   # precomputed ray angles
            if valid:
                hit_positions[i, :] = np.asarray(hit_pos, dtype=float)

        # Sector edges (relative to drone yaw)
        sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1) - np.pi / n_sectors

        sector_labels = [f'sector_{i}' for i in range(n_sectors)]

        features = []
        # Compute features per sector
        for j in range(n_sectors):
            a0, a1 = sector_edges[j], sector_edges[j + 1]
            if j==0:
                in_sector = (angles >= sector_edges[n_sectors]) | (angles < sector_edges[j+1])
            else:
                in_sector = (angles >= a0) & (angles < a1)
            beams_in_sector = in_sector.sum()

            if beams_in_sector == 0:
                features.extend([1.0, 1.0, 0.0, 0.0])
                continue

            valid_idx = in_sector & mask
            cnt_hits = valid_idx.sum()
            hit_density_sector = cnt_hits / beams_in_sector
            if cnt_hits > 0:
                rsec = ranges[valid_idx]
                rmin = rsec.min() / max_range
                rmean = rsec.mean() / max_range

                # LOS angle from the closest hit in this sector
                hit_global_indices = np.where(valid_idx)[0]
                hit_ranges = ranges[valid_idx]
                k_global = int(hit_global_indices[np.argmin(hit_ranges)])

                hit_vec = hit_positions[k_global, :] - drone_pos
                bearing_global = float(np.arctan2(hit_vec[1], hit_vec[0]))
                bearing_rel = _wrap_to_pi(bearing_global - yaw)
                los_angle = bearing_rel / np.pi  # normalize to [-1, 1]
                # if self.DRONE_ID == 0:
                #     print(f"Sector '{sector_labels[j]}' has {cnt_hits} hit(s): min={rmin:.2f}, mean={rmean:.2f}, fraction={hit_density_sector:.2f}")
            else:
                rmin = 1.0
                rmean = 1.0
                los_angle = 0.0  # neutral when no hit

            features.extend([rmin, rmean, hit_density_sector, float(los_angle)])

        return np.array(features, dtype=float)



    
    def _processDetection(self, obstacle_data):
        """
        Screen obstacle_data based on the detection from self.DETECTED_OBS_IDS. 

        Args:
            obstacle_data (dict): dictionary of dictionary of obstacles data where
                key is the obstacle's id and values include 'position' and 'size'.

        Returns:
            None.

        """
        if len(self.DETECTED_OBS_IDS) != 0:
            tempObs = []
            for j in self.DETECTED_OBS_IDS:
                if str(j) in obstacle_data:
                    self.DETECTED_OBS_DATA[str(j)] = {"position": obstacle_data[str(j)]["position"],
                                                          "size": obstacle_data[str(j)]["size"]}
                    tempObs.append(obstacle_data[str(j)]["position"])
        else:
            self.DETECTED_OBS_DATA = {}

    def _generateWaypoints(self, home_pos, destination, num_waypoints):
        """
        Generate waypoints in 3D for a straight line from home_pos to destination.

        Parameters:
        - home_pos: tuple of floats (x, y, z) representing the starting position.
        - destination: tuple of floats (x, y, z) representing the destination position.
        - num_waypoints: integer, number of waypoints to generate along the path.

        Returns:
        - waypoints: numpy array of shape (3, num_waypoints) where each row is x, y, z coordinates.
        """
        # Generate an array of points from 0 to 1 with num_waypoints
        t_values = np.linspace(0, 1, num_waypoints+2)

        # Interpolate each dimension and stack as rows in a (3, num_waypoints) array
        waypoints = np.array([
            (1 - t_values) * home_pos[0] + t_values * destination[0],
            (1 - t_values) * home_pos[1] + t_values * destination[1],
            (1 - t_values) * home_pos[2] + t_values * destination[2]
        ])
        # Remove the first column to offset the path
        waypoints = waypoints[:, 2:]  # Shape becomes (3, num_waypoints)

        return waypoints
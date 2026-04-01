import warnings
import numpy as np
import math
import random

class RouteMission:
    def __init__(self, numDrones=1, scenario=1, timeLimit=None):
        self.NUM_DRONES = numDrones
        self.SCENE = scenario
        self.TIME_LIMIT = timeLimit
        self.INIT_XYZS = np.zeros(3)
        self.INIT_RPYS = np.zeros(3)
        self.DESTINS = np.zeros(3)
        self.WAYPOINTS = []

    def generateRandomMission(self, maxNumDrone, minNumDrone=1, seed=None,
                               min_distance=20.0, max_attempts=100,
                               min_travel_distance=40, mission_cfg=None):
        """Generate a random mission with drones spaced apart and traveling a minimum distance.

        Parameters
        ----------
        mission_cfg : dict, optional
            Override default area / spacing constants.  Recognised keys:
            origin, base_r, base_r_d, h_step, radius_variation,
            angle_variation_deg, z_variation, min_distance,
            min_travel_distance.
        """
        _mc = mission_cfg or {}
        min_distance = _mc.get("min_distance", min_distance)
        min_travel_distance = _mc.get("min_travel_distance", min_travel_distance)

        self._validate_random_mission_inputs(minNumDrone, maxNumDrone, seed, min_travel_distance)

        ORIGIN = _mc.get("origin", [0, 0, 5])
        BASE_R  = _mc.get("base_r", 60)
        BASE_R_D = _mc.get("base_r_d", 50)
        H_STEP = _mc.get("h_step", 0)
        RADIUS_VARIATION = _mc.get("radius_variation", 0.5)
        ANGLE_VARIATION = _mc.get("angle_variation_deg", 90) * np.pi / 180
        Z_VARIATION = _mc.get("z_variation", 0)

        retry_seed = seed
        max_retries = 200
        for _attempt in range(max_retries):
            self._set_random_seed(retry_seed)
            self.NUM_DRONES = random.randint(minNumDrone, maxNumDrone)

            inits, dests, waypoints = self._generate_drone_positions(
                self.NUM_DRONES, ORIGIN, BASE_R, BASE_R_D, H_STEP,
                RADIUS_VARIATION, ANGLE_VARIATION, Z_VARIATION,
                min_distance, max_attempts, min_travel_distance
            )

            if self._verify_distances(inits, dests, min_distance, self.NUM_DRONES, min_travel_distance):
                break
            if retry_seed is not None:
                retry_seed += 1
        else:
            warnings.warn(
                f"generateRandomMission: could not satisfy constraints after "
                f"{max_retries} retries (min_dist={min_distance}, "
                f"min_travel={min_travel_distance}, n_drones={self.NUM_DRONES}, "
                f"base_r={BASE_R}, base_r_d={BASE_R_D}). "
                f"Using best-effort positions — constraints may be violated.",
                RuntimeWarning,
            )

        self._set_mission_parameters(inits, dests, waypoints)

    def _generate_drone_positions(self, num_drones, origin, base_r, base_r_d, h_step,
                                  radius_var, angle_var, z_var, min_distance, max_attempts, min_travel_distance):
        inits = np.zeros((num_drones, 3))
        rpys = np.zeros((num_drones, 3))
        dests = np.zeros((num_drones, 3))
        waypoints = []

        init_used, dest_used = [], []

        for i in range(num_drones):
            init_base = [origin[0], origin[1], origin[2]]
            init = self._generate_position_with_avoidance(
                init_base, base_r, init_used, min_distance, max_attempts,
                0, i, num_drones, radius_var, angle_var, z_var, h_step
            )
            inits[i] = init
            init_used.append(init)

            dest = self._generate_valid_destination(
                origin, base_r_d, dest_used, min_distance, max_attempts,
                init, min_travel_distance, i, num_drones, radius_var, angle_var, z_var, h_step
            )
            dests[i] = dest
            dest_used.append(dest)

            yaw = math.atan2(dest[1] - init[1], dest[0] - init[0])
            rpys[i] = [0, 0, yaw-np.pi/2]
            waypoints.append(np.vstack((init, dest)))

        return inits, dests, waypoints

    def _generate_valid_destination(self, origin, base_r, existing, min_dist, max_attempts,
                                    init_pos, min_travel_dist, i, total, r_var, a_var, z_var, h_step):
        dest_base = [origin[0], origin[1], origin[2]]
        for _ in range(max_attempts):
            dest = self._generate_position_with_avoidance(
                dest_base, base_r, existing, min_dist, max_attempts,
                np.pi / 2, i, total, r_var, a_var, z_var, h_step
            )
            if self._calculate_distance(dest, init_pos) >= min_travel_dist:
                return dest
        # print(f" Fallback destination for drone {i}")
        # input("Press Enter to continue. . .")
        return dest

    def _generate_position_with_avoidance(self, base, radius, existing, min_dist, max_attempts,
                                          angle_offset, i, total, r_var, a_var, z_var, h_step):
        for _ in range(max_attempts):
            r = radius + random.uniform(-r_var, r_var)
            base_angle = (i / total) * 2 * np.pi + np.pi / 2 + angle_offset
            angle = base_angle + random.uniform(-a_var, a_var)
            pos = [base[0] + r * np.cos(angle),
                   base[1] + r * np.sin(angle),
                   base[2] + i * h_step + random.uniform(-z_var, z_var)]
            if self._is_position_valid(pos, existing, min_dist):
                return pos

        # print(f" Fallback initial position for drone {i}")
        fallback_r = max(radius, min_dist * total / (2 * np.pi) * 1.2)
        fallback_angle = (i / total) * 2 * np.pi + angle_offset
        return [base[0] + fallback_r * np.cos(fallback_angle),
                base[1] + fallback_r * np.sin(fallback_angle),
                base[2] + i * h_step]

    def _verify_distances(self, inits, dests, min_dist, num, min_travel_dist):
        def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))

        min_init_dist = float('inf')
        min_dest_dist = float('inf')
        min_travel_actual = float('inf')

        for i in range(num):
            min_travel_actual = min(min_travel_actual, dist(inits[i], dests[i]))
            for j in range(i + 1, num):
                min_init_dist = min(min_init_dist, dist(inits[i], inits[j]))
                min_dest_dist = min(min_dest_dist, dist(dests[i], dests[j]))

        # print(f"Initial min dist: {min_init_dist:.2f}, Destination min dist: {min_dest_dist:.2f}, Travel min: {min_travel_actual:.2f}")

        if num == 1:
            return min_travel_actual >= min_travel_dist
        return min_init_dist >= min_dist and min_dest_dist >= min_dist and min_travel_actual >= min_travel_dist

    def _set_mission_parameters(self, init, dest, waypoints):
        rpys = np.zeros((self.NUM_DRONES, 3))
        # for i in range(self.NUM_DRONES):
        #     dx, dy = dest[i][0] - init[i][0], dest[i][1] - init[i][1]
        #     rpys[i] = [0, 0, math.atan2(dy, dx)]
 
        self.INIT_XYZS = init
        self.INIT_RPYS = rpys
        self.DESTINS = dest
        self.WAYPOINTS = waypoints

    def _calculate_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _is_position_valid(self, new, existing, min_dist):
        return all(self._calculate_distance(new, ex) >= min_dist for ex in existing)

    def _set_random_seed(self, seed):
        if seed is not None:
            # print(f"Seed: {seed}")
            random.seed(seed)
            np.random.seed(seed)

    def _validate_random_mission_inputs(self, min_drones, max_drones, seed, min_travel):
        if not (isinstance(min_drones, int) and isinstance(max_drones, int)):
            raise TypeError("minNumDrone and maxNumDrone must be integers")
        if min_drones < 0 or max_drones < 0:
            raise ValueError("Drone counts must be non-negative")
        if min_drones > max_drones:
            raise ValueError("minNumDrone cannot be greater than maxNumDrone")
        if seed is not None and not isinstance(seed, int):
            raise TypeError("Seed must be an integer or None")
        if min_travel < 0:
            raise ValueError("min_travel_distance must be non-negative")
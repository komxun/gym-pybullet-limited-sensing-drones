import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.utils.utils import nnlsRPM

class PIDVelocityControl(BaseControl):
    """Simple velocity-only PID controller for DroneModel.HB"""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        # if self.DRONE_MODEL != DroneModel.HB:
        #     print("[ERROR] PIDVelocityControl requires DroneModel.HB")
        #     exit()

        # PID gains for velocity control
        self.P_COEFF_VEL = np.array([0.3, 0.3, 0.5])
        self.I_COEFF_VEL = np.array([0.05, 0.05, 0.1])
        self.D_COEFF_VEL = np.array([0.02, 0.02, 0.03])

        self.P_COEFF_TOR = np.array([0.18, 0.18, 0.18])
        self.I_COEFF_TOR = np.array([0.018, 0.018, 0.018])
        self.D_COEFF_TOR = np.array([0.002, 0.002, 0.002])

        self.MAX_ROLL_PITCH = np.pi / 6
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = 4 * self.KF * self.MAX_RPM ** 2
        self.MAX_XY_TORQUE = self.L * self.KF * self.MAX_RPM ** 2
        self.MAX_Z_TORQUE = 2 * self.KM * self.MAX_RPM ** 2

        self.A = np.array([
            [1, 1, 1, 1],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [-1, 1, -1, 1]
        ])
        self.INV_A = np.linalg.inv(self.A)
        self.B_COEFF = np.array([1 / self.KF, 1 / (self.KF * self.L), 1 / (self.KF * self.L), 1 / self.KM])
        self.reset()

    def reset(self):
        super().reset()
        self.last_vel_e = np.zeros(3)
        self.integral_vel_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_vel,
                       target_rpy=np.zeros(3)):
        """Compute motor RPMs based only on velocity error."""
        self.control_counter += 1
        target_yaw = target_rpy[2]
        thrust, computed_target_rpy, vel_e = self._velocityPID(control_timestep,
                                                               cur_vel,
                                                               cur_quat,
                                                               target_vel)

        rpm = self._attitudePID(control_timestep, thrust, cur_quat, computed_target_rpy)
        # rpm = self._attitudePID(control_timestep, thrust, cur_quat, computed_target_rpy)

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, vel_e, computed_target_rpy[2] - cur_rpy[2]

    def _velocityPID(self, control_timestep, cur_vel, cur_quat, target_vel):
        """PID velocity control → target thrust and attitude."""
        vel_e = target_vel - np.array(cur_vel).reshape(3)
        d_vel_e = (vel_e - self.last_vel_e) / control_timestep
        self.last_vel_e = vel_e
        self.integral_vel_e += vel_e * control_timestep

        # Compute desired total force vector in world frame
        target_force = np.array([0, 0, self.GRAVITY]) \
                       + np.multiply(self.P_COEFF_VEL, vel_e) \
                       + np.multiply(self.I_COEFF_VEL, self.integral_vel_e) \
                       + np.multiply(self.D_COEFF_VEL, d_vel_e)

        # Compute target roll/pitch to produce that force
        target_rpy = np.zeros(3)
        sign_z = np.sign(target_force[2]) or 1
        target_rpy[0] = np.arcsin(-sign_z * target_force[1] / np.linalg.norm(target_force))
        target_rpy[1] = np.arctan2(sign_z * target_force[0], sign_z * target_force[2])
        target_rpy[2] = 0.0
        

        target_rpy[0] = np.clip(target_rpy[0], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        target_rpy[1] = np.clip(target_rpy[1], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)

        # Transform to body frame and get thrust along z-axis
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        thrust = np.dot(cur_rotation, target_force)

        return thrust[2], target_rpy, vel_e

    def _attitudePID(self, control_timestep, thrust, cur_quat, target_rpy):
        """Attitude PID loop (yaw fixed to 0)."""
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        rpy_e = target_rpy - np.array(cur_rpy).reshape(3,)
        rpy_e[2] = (rpy_e[2] + np.pi) % (2 * np.pi) - np.pi

        d_rpy_e = (rpy_e - self.last_rpy_e) / control_timestep
        self.last_rpy_e = rpy_e
        self.integral_rpy_e += rpy_e * control_timestep

        target_torques = np.multiply(self.P_COEFF_TOR, rpy_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e) \
                         + np.multiply(self.D_COEFF_TOR, d_rpy_e)

        return nnlsRPM(thrust=thrust,
                       x_torque=target_torques[0],
                       y_torque=target_torques[1],
                       z_torque=target_torques[2],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True)
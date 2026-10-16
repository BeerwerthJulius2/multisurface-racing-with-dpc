from copy import copy

import yaml

from external.nmpc.Model_Predictive_Controller.Nominal_NMPC.NMPC_class import Nonlinear_Model_Predictive_Controller

import numpy as np


from external.dpc.dataset.data_processing.vehicle_data_processing import wrap_angle_pi
from external.dpc.util.utils import nearest_point


class NMPCAdapter:
    def __init__(self, initial_state, waypoints):
        self.update_waypoints(waypoints)

        self.control_freq_hz = 10.
        self.prediction_horizon = 11
        self.last_acc = 0.0

        config_path = "external/nmpc/Config/"

        sim_main_params_file = "RoboRacer-Multibody/sim_main_params.yaml"
        MPC_params_file = "RoboRacer-Multibody/MPC_params.yaml"
        with open(config_path + sim_main_params_file, 'r') as file:
            sim_main_params = yaml.load(file, Loader=yaml.FullLoader)

        self.vehicle_nmpc = Nonlinear_Model_Predictive_Controller(config_path, MPC_params_file, sim_main_params, initial_state)

    def plan(self, states, last_acc, waypoints=None):
        if waypoints is not None:
            self.update_waypoints(waypoints)

        ref, _ = self.get_reference(states, pred_h=self.prediction_horizon)
        orientation_x, orientation_y, orientation_z, orientation_w = self.yaw_to_quaternion(ref[3])
        reference_traj = {'pos_x':ref[0],
                          'pos_y':ref[1],
                          'orientation_x':orientation_x,
                          'orientation_y':orientation_y,
                          'orientation_z':orientation_z,
                          'orientation_w':orientation_w,
                          'ref_v':ref[2],
                          'ref_acc':ref[4],
                          'ref_yaw':ref[3]}

        initial_state = np.array([
            0.0,  # x
            0.0,  # y
            0.0,  # yaw
            states[2],  # vx
            states[4],  # vy
            states[5],  # yaw_rate
            states[6],  # steering
        ])
        initial_state = np.append(initial_state, last_acc)
        self.vehicle_nmpc.set_initial_state(initial_state)

        u, prediction, _ = self.vehicle_nmpc.solve(reference_traj)

        yaw0 = states[3]

        # rotation back to global
        R = np.array([
            [np.cos(yaw0), -np.sin(yaw0)],
            [np.sin(yaw0), np.cos(yaw0)]
        ])

        # reference
        ref_global = R @ np.vstack((ref[0], ref[1]))
        mpc_ref_path_x = ref_global[0] + states[0]
        mpc_ref_path_y = ref_global[1] + states[1]

        # prediction
        pred_global = R @ np.vstack((prediction[:, 0], prediction[:, 1]))
        pred_x = pred_global[0] + states[0]
        pred_y = pred_global[1] + states[1]

        dt = 1.0 / float(self.control_freq_hz)
        u[0] = last_acc + u[0] * dt


        return u, mpc_ref_path_x, mpc_ref_path_y, pred_x, pred_y, pred_x, pred_y

    def update_waypoints(self, waypoints):
        self.waypoints = waypoints
        self._progress = waypoints[:, 0]
        self._ref_x = waypoints[:, 1]
        self._ref_y = waypoints[:, 2]
        self._ref_speed = waypoints[:, 5]
        self._ref_yaw = waypoints[:, 3]
        self._ref_xy = np.column_stack([self._ref_x, self._ref_y])

    def get_reference(self, last_output, pred_h):

        x = float(last_output[0])
        y = float(last_output[1])
        v = float(last_output[2])

        _, _, t, ind = nearest_point(np.array([x, y]), self._ref_xy)

        n = self._progress.shape[0]

        # --------------------------------------------------
        # exact arc-length position on raceline
        # --------------------------------------------------

        s0 = self._progress[ind] + t * (self._progress[ind + 1] - self._progress[ind])

        dt = 1.0 / float(self.control_freq_hz)

        #s_future = s0 + np.cumsum(np.ones(pred_h) * abs(v) * dt)
        s_future = s0 + np.arange(pred_h) * abs(v) * dt

        track_len = self._progress[-1]
        s_future = np.mod(s_future, track_len)

        # --------------------------------------------------
        # find raceline segments
        # --------------------------------------------------

        idx = np.searchsorted(self._progress, s_future, side="right") - 1
        idx = np.clip(idx, 0, n - 2)

        s_i = self._progress[idx]
        s_ip1 = self._progress[idx + 1]

        tau = (s_future - s_i) / np.maximum(s_ip1 - s_i, 1e-9)

        # --------------------------------------------------
        # interpolate
        # --------------------------------------------------

        ref = np.zeros((5, pred_h), dtype=float)

        ref[0] = self._ref_x[idx] + tau * (self._ref_x[idx + 1] - self._ref_x[idx])
        ref[1] = self._ref_y[idx] + tau * (self._ref_y[idx + 1] - self._ref_y[idx])
        ref[2] = self._ref_speed[idx] + tau * (self._ref_speed[idx + 1] - self._ref_speed[idx])

        yaw_i = self._ref_yaw[idx]
        yaw_ip1 = self._ref_yaw[idx + 1]

        dyaw = np.arctan2(np.sin(yaw_ip1 - yaw_i), np.cos(yaw_ip1 - yaw_i))
        ref[3] = np.mod(yaw_i + tau * dyaw, 2 * np.pi)


        # first step uses current speed
        ref[4,0] = (ref[2,0] - v) / dt

        # remaining steps from speed differences
        ref[4,1:] = np.diff(ref[2]) / dt

        yaw0 = last_output[3]

        # --- position to local frame ---
        dx = ref[0] - x
        dy = ref[1] - y

        ref[0] = np.cos(yaw0) * dx + np.sin(yaw0) * dy
        ref[1] = -np.sin(yaw0) * dx + np.cos(yaw0) * dy

        # --- yaw to local frame ---
        ref[3] = np.arctan2(
            np.sin(ref[3] - yaw0),
            np.cos(ref[3] - yaw0)
        )

        return ref, s0

    def yaw_to_quaternion(self, yaw: np.ndarray):
        """
        Convert a yaw angle trajectory to quaternion orientation arrays
        compatible with the TUM-CONTROL reference trajectory format.

        Parameters
        ----------
        yaw : np.ndarray
            Array of yaw angles (rad).

        Returns
        -------
        dict
            Dictionary containing quaternion components.
        """

        yaw = np.asarray(yaw)

        orientation_x = np.zeros_like(yaw)
        orientation_y = np.zeros_like(yaw)
        orientation_z = np.sin(yaw / 2.0)
        orientation_w = np.cos(yaw / 2.0)

        return orientation_x, orientation_y, orientation_z, orientation_w
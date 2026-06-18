import os

import numpy as np
from hydra import compose, initialize_config_dir

import external.dpc.experiment as experiment
from external.dpc.dataset.data_processing.vehicle_data_processing import wrap_angle_pi
from external.dpc.experiment_logging.experiment_logger import ExperimentLogger
from external.dpc.dataset.trajectory_dataset import TrajectoryDataset
from external.dpc.controller.vehicle_dpc.vehicle_dpc import VehicleDPC
from external.dpc.util.utils import nearest_point


class DPCAdapter:
    def __init__(self, waypoints):
        config_dir = os.path.join(os.path.dirname(experiment.__file__), "config")

        data_dir = os.path.join(
            os.getcwd(), "data/dpc_experiments/02_tracking")

        print(f"Loading Hydra config from: {config_dir}")
        print(f"Using data directory: {data_dir}")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "+experiment=vehicle_dpc_eval_multisurface",
                    f"dataset.path={data_dir}",
                ]
            )

        self.experiment_logger = ExperimentLogger()
        self.data = TrajectoryDataset(cfg)
        self.vehicle_dpc = VehicleDPC(data=self.data, cfg=cfg, data_logger=self.experiment_logger)

        self.cfg = cfg
        self.update_waypoints(waypoints)

    def plan(self, initial_inputs, initial_outputs, waypoints=None):
        if waypoints is not None:
            self.update_waypoints(waypoints)

        last_output = initial_outputs[-1]
        initial_inputs = np.stack(list(initial_inputs), axis=1)  # [accl, delta]
        initial_outputs = np.stack(list(initial_outputs), axis=1)  # [x, y, v, yaw]

        ref, _ = self.get_reference(last_output, pred_h=self.cfg.controller.prediction_horizon)
        accl_seq, delta_seq, pred_x, pred_y, _, _, meta = self.vehicle_dpc.plan(
            initial_inputs=initial_inputs,
            initial_outputs=initial_outputs,
            reference_trajectory=ref,
        )

        mpc_ref_path_x = ref[0]
        mpc_ref_path_y = ref[1]

        u = [accl_seq[0], delta_seq[0]]

        return u, mpc_ref_path_x, mpc_ref_path_y, pred_x, pred_y, pred_x, pred_y, meta

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

        dt = 1.0 / float(self.cfg.experiment.control_freq_hz)

        s_future = s0 + np.cumsum(np.ones(pred_h) * abs(v) * dt)

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

        ref = np.zeros((4, pred_h), dtype=float)

        ref[0] = self._ref_x[idx] + tau * (self._ref_x[idx + 1] - self._ref_x[idx])
        ref[1] = self._ref_y[idx] + tau * (self._ref_y[idx + 1] - self._ref_y[idx])
        ref[2] = self._ref_speed[idx] + tau * (self._ref_speed[idx + 1] - self._ref_speed[idx])

        yaw_i = wrap_angle_pi(self._ref_yaw[idx])
        yaw_ip1 = wrap_angle_pi(self._ref_yaw[idx + 1])

        dyaw = wrap_angle_pi(yaw_ip1 - yaw_i)
        ref[3] = wrap_angle_pi(yaw_i + tau * dyaw)

        return ref, s0
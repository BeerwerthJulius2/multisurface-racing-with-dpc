"""Grip usage of the simulation datasets (Section IV-A): share of the samples whose lateral acceleration v * d(yaw)/dt exceeds
70% of the available grip mu * g, for the raceline data (02_tracking_v2, closed loop) and the random-excitation data
(01_prediction_v2, Fig. 1). Rows are 0.1 s apart; columns t, agent id, accel, steering, x, y, v, yaw.
usage (repo root): python evaluation/01_prediction/grip_usage.py"""
import glob
import numpy as np

SETS = [("raceline", 0.5, "data/dpc_experiments/02_tracking_v2/pure_pursuit_0_5/inputs_outputs_0.5.npy"),
        ("raceline", 1.1, "data/dpc_experiments/02_tracking_v2/pure_pursuit_1_1/inputs_outputs_1.1.npy"),
        ("random excitation", 0.5, "data/dpc_experiments/01_prediction_v2/random_0_5/inputs_outputs_random_0.5.npy"),
        ("random excitation", 1.1, "data/dpc_experiments/01_prediction_v2/random_1_1/inputs_outputs_random_1.1.npy")]
DT, G = 0.1, 9.81
for name, mu, path in SETS:
    a = np.load(glob.glob(path)[0])
    agent, v, yaw = a[:, 1], a[:, 6], a[:, 7]
    lat = np.concatenate([np.abs(v[agent == k][1:] * np.diff(np.unwrap(yaw[agent == k])) / DT) for k in np.unique(agent)])
    print(f"{name:18s} mu {mu}: median grip use {np.median(lat) / (mu * G) * 100:3.0f}%, "
          f"samples above 70% of the grip {np.mean(lat > 0.7 * mu * G) * 100:3.0f}%")

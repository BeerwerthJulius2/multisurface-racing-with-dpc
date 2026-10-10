"""Offline closed-loop proxy: run the real DPC controller on recorded windows, no simulator.

For a recorded window (inputs/outputs of the last T_INI steps), the controller plans with the recorded future
outputs as the reference: exactly the sampling + QP that runs online (VehicleDPC.plan), only the plant is missing.
The query's time neighbourhood is blocked in both recordings (they follow the same reference in sync), so the
controller has to work from other laps; this is done on the segment cache, the library is unchanged.

Reported per setting:
  fail      share of windows where the QP did not return a usable solution
  plan      RMS planned position error vs the reference over the horizon [m]
  dv        RMS planned speed error [m/s]
  sel/use   share of the selected columns / of the QP usage from the matching friction (corners)

Usage:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python evaluation/exploratory/segment_sampling/offline/offline_qp_replay.py
"""
import logging
import os
import sys

import numpy as np
from hydra import compose, initialize_config_dir

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
import external.dpc.experiment as experiment  # noqa: E402
from external.dpc.dataset.trajectory_dataset import TrajectoryDataset  # noqa: E402
from external.dpc.controller.vehicle_dpc.vehicle_dpc import VehicleDPC  # noqa: E402

RAW = {0: "data/dpc_experiments/02_tracking/pure_pursuit_0_5/inputs_outputs_0.5.npy",
       1: "data/dpc_experiments/02_tracking/pure_pursuit_1_1/inputs_outputs_1.1.npy"}
T_INI, N = 5, 10
EXCLUDE = 30
N_QUERY = 200  # per recording
SETTINGS = [(cost, G, k) for cost in ("fit", "qp") for G, k in [(25, 4), (25, 8), (40, 6), (40, 8), (50, 2), (50, 3)]]


class NoLog:
    """The controller's step logger keeps every matrix; not needed here."""
    def __getattr__(self, name):
        return lambda *a, **kw: None


def build(cost, G, k):
    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(experiment.__file__), "config"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "+experiment=vehicle_dpc_eval_multisurface",
            f"dataset.path={os.path.join(REPO_ROOT, 'data/dpc_experiments/02_tracking')}",
            "++dataset.reuse_cached_merge=true",
            "controller.path_sampling.sampling=segment",
            f"++controller.path_sampling.segment_length={G}",
            f"++controller.path_sampling.segment_cost={cost}",
            f"controller.num_traj={G * k}",
        ])
    data = TrajectoryDataset(cfg)
    return data, VehicleDPC(data=data, cfg=cfg, data_logger=NoLog())


def main():
    logging.disable(logging.WARNING)
    raw = {lab: np.load(p) for lab, p in RAW.items()}
    lat = {lab: np.abs(a[:, 6] * np.gradient(np.unwrap(a[:, 7]), 0.1)) for lab, a in raw.items()}
    threshold = np.percentile(np.concatenate(list(lat.values())), 80)
    rng = np.random.default_rng(0)
    queries = [(lab, j) for lab, a in raw.items()
               for j in rng.choice(np.arange(EXCLUDE, len(a) - T_INI - N - EXCLUDE), N_QUERY, replace=False)]

    print(f"{N_QUERY} recorded windows per friction, reference = recorded future\n")
    print(f"{'cost':>4} {'G':>3} {'k':>2} {'cols':>4} | {'fail':>5} {'plan [m]':>8} {'dv':>6} | "
          f"{'corners: plan':>13} {'sel':>5} {'use':>5}")
    for cost, G, k in SETTINGS:
        data, ctrl = build(cost, G, k)
        labels = np.asarray(data.dataset_labels)
        run_start = {lab: np.flatnonzero(labels == lab)[0] for lab in np.unique(labels)}
        index_in_run = np.arange(len(labels)) - np.array([run_start[l] for l in labels])
        a0 = raw[0]
        ctrl.plan(a0[:T_INI, 2:4].T, a0[:T_INI, 4:8].T, a0[T_INI:T_INI + N, 4:8].T)  # builds the segment cache
        cache = data._segment_cache
        b_free = cache["b"].copy()
        starts = cache["starts"]

        rows = []
        for lab, j in queries:
            a = raw[lab]
            excluded = np.abs(index_in_run - j) <= EXCLUDE
            blocked = np.concatenate([[0], np.cumsum(excluded)])
            cache["b"] = b_free.copy()
            cache["b"][blocked[starts + G] != blocked[starts]] += 1e6  # never selected
            ref = a[j + T_INI:j + T_INI + N, 4:8].T
            try:
                accl, steer, x, y, v, yaw, meta = ctrl.plan(a[j:j + T_INI, 2:4].T, a[j:j + T_INI, 4:8].T, ref)
                plan = np.sqrt(np.mean((np.asarray(x) - ref[0]) ** 2 + (np.asarray(y) - ref[1]) ** 2))
                dv = np.sqrt(np.mean((np.asarray(v) - ref[2]) ** 2))
                ul, uf, df = np.asarray(meta["unique_labels"]), np.asarray(meta["usage_fraction"]), np.asarray(meta["data_fraction"])
                use, sel = uf[ul == lab].sum(), df[ul == lab].sum()
                rows.append((0, plan, dv, sel, use, lat[lab][j + T_INI - 1] > threshold))
            except Exception:
                rows.append((1, np.nan, np.nan, np.nan, np.nan, lat[lab][j + T_INI - 1] > threshold))
        cache["b"] = b_free
        r = np.array(rows, float)
        c = r[:, 5].astype(bool)
        print(f"{cost:>4} {G:>3} {k:>2} {G * k:>4} | {r[:, 0].mean():>5.1%} {np.nanmedian(r[:, 1]):>8.3f} "
              f"{np.nanmedian(r[:, 2]):>6.3f} | {np.nanmedian(r[c, 1]):>13.3f} {np.nanmean(r[c, 3]):>5.0%} "
              f"{np.nanmean(r[c, 4]):>5.0%}", flush=True)


if __name__ == "__main__":
    main()

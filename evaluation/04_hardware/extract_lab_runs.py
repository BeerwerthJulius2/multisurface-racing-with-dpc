"""Per-run results of the CPM Lab evaluation (2026-10-15, 18 runs: 2 tyre sets x pooled / oracle / inverse x 3) from the lab
logs of the ros2-workspace-lab-dpc workspace -> hardware_runs.json (committed; the raw logs, 77 MB, stay in the lab workspace).
Lateral error: rms distance to the raceline (and |distance| every 4th step); lap time from the mean progress speed (and the
time of each complete lap); prediction error: position error of the controller's prediction vs the measured trajectory, rms
over the horizon (and at its last step), median over the run (and every 4th step).
Tyre set by time (slippery block before 12:12, then normal); arm from controller.path_sampling.oracle_labels.
usage (lab environment: ROS 2 jazzy, the dpc venv): extract_lab_runs.py <workspace> [out.json]"""
import sys, glob, os, json, numpy as np, logging, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
WS = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "hardware_runs.json")
sys.path = [p for p in sys.path if "src/deepc" not in p]; sys.path.insert(0, WS + "/src/deepc")
from omegaconf import OmegaConf, open_dict
from dataset.trajectory_dataset import TrajectoryDataset
from controller.vehicle_dpc.vehicle_dpc import VehicleDPC
sys.path.insert(0, WS + "/src/dpc-controller/src"); from raceline import load_raceline, project
out = []; ctl = None
for d in sorted(glob.glob(WS + "/src/dpc-controller/log/2026-10-15T1[12]-*")):
    run = os.path.basename(d); tyre = "slippery" if run[11:16] < "12-12" else "normal"
    cfg = OmegaConf.load(d + "/vehicle_1/config.yaml"); cr = cfg.controller
    orc = cr.path_sampling.get("oracle_labels"); data = "pooled" if orc is None else ("slippery" if 7 in list(orc) else "normal")
    arm = "pooled" if data == "pooled" else ("oracle" if data == tyre else "inverse")
    z = np.load(glob.glob(d + "/vehicle_1/results_*.npz")[0]); t = z["experiment/t"] - z["experiment/t"][0]; st = z["experiment/state"].astype(float)
    rl = load_raceline(WS + "/src/dpc-controller/config/" + cr.raceline_file, cr.raceline_a_lat); li = None; lat = []; s_un = []; acc = 0; sp = None
    for k in range(len(t)):
        s0, li, dd = project(rl, st[k, 0], st[k, 1], li); lat.append(dd)
        if sp is not None:
            ds = s0 - sp; ds += rl["length"] if ds < -rl["length"] / 2 else 0; acc += ds
        sp = s0; s_un.append(acc)
    lat, s_un = np.array(lat), np.array(s_un); m = (t > 3) & (t < t[-1] - 0.5)   # after the rolling start, before the stop
    lap = rl["length"] / ((s_un[m][-1] - s_un[m][0]) / (t[m][-1] - t[m][0]))
    k0 = np.ceil(s_un[m][0] / rl["length"]); marks = rl["length"] * np.arange(k0, np.floor(s_un[m][-1] / rl["length"]) + 1)
    t_marks = np.interp(marks, s_un[m], t[m]); laps = np.diff(t_marks)               # complete laps inside the window
    if ctl is None:
        with open_dict(cfg): cfg.dataset = cfg.controller.dataset; cfg.experiment.simulation_freq_hz = None
        class L:
            def log_dpc_step(self, *a, **k): pass
        ctl = VehicleDPC(data=TrajectoryDataset(cfg), cfg=cfg, data_logger=L())
    sd = np.asarray(ctl.output_std, float); H = cr.prediction_horizon; YF = z["dpc/future_outputs"]; e_rms, e_end = [], []
    for i in np.where(m)[0][::4]:
        if i + H >= len(t): continue
        y_true = ctl.normalize_fcn(ctl.align_fcn(st[i + 1:i + H + 1].T.copy(), st[i].copy()), ctl.output_mean, ctl.output_std, 4)
        er = (YF[i].astype(float) - y_true) * sd[:, None]; ep = np.hypot(er[0], er[1])
        e_rms.append(np.sqrt(np.mean(ep ** 2))); e_end.append(ep[-1])
    out.append(dict(run=run, tyre=tyre, arm=arm, lat_cm=100 * float(np.sqrt(np.mean(lat[m] ** 2))), lap_s=float(lap),
                    pred_rms_cm=100 * float(np.median(e_rms)), pred_end_cm=100 * float(np.median(e_end)), horizon=int(H),
                    lap_times_s=[round(float(x), 3) for x in laps], lat_steps_cm=[round(100 * abs(float(x)), 2) for x in lat[m][::4]],
                    pred_steps_cm=[round(100 * float(x), 2) for x in e_rms]))
    print({k: v for k, v in out[-1].items() if not isinstance(v, list)}, len(laps), 'laps')
json.dump(out, open(OUT, "w"), indent=1); print("written", OUT)

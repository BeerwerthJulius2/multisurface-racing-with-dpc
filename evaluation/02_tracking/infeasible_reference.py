"""Numbers of the infeasible-reference check (run_infeasible_reference.sh): per run whether the laps were completed, where the car
left the track, and the median prediction error (rms position over the horizon, cm) on the first 60 m, i.e. before the first
corner at mu 0.5 ends the 1.1x runs. usage (repo root): python evaluation/02_tracking/infeasible_reference.py"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from helpers.eval_metrics import prediction_error   # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "infeasible_reference")
print(f"{'run':24s} {'laps done':>9s} {'reached [m]':>11s} {'pred. error < 60 m [cm]':>24s} {'pred. error all [cm]':>21s}")
for f in sorted(glob.glob(os.path.join(D, "log_dpc_eval_l_shape_inf_*"))):
    d = json.load(open(f)); nan = lambda k: np.array([np.nan if v is None else v for v in d[k]], float)
    p, te, lap = nan("track_progress"), nan("tracking_error"), nan("lap_n"); pe = prediction_error(d["pred_xy"], nan("x"), nan("y"))
    done = lap.max() >= 2 and np.nanmax(te) < 5
    m60 = (p < 60) & np.isfinite(pe) & (lap == 0)
    print(f"{os.path.basename(f)[25:]:24s} {str(bool(done)):>9s} {('all' if done else '%.0f' % np.nanmax(p)):>11s} "
          f"{100 * np.nanmedian(pe[m60]):24.1f} {100 * np.nanmedian(pe[np.isfinite(pe)]):21.1f}")

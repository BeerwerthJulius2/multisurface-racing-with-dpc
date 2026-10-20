"""Prediction error of the DPC predictor on the paper's dataset (data/dpc_experiments/02_tracking_v2, mu 0.5 and 1.1) for held-out
windows: with 300 contextually sampled columns (the controller's setting) from the matched dataset, from the mismatched one and
from the pooled dataset, with 300 random columns of the pooled dataset, and with the full datasets (matched / mismatched / pooled,
no sampling). The predictor
is the controller's regularised DPC predictor with the future inputs fixed (helpers/implied_friction.ls_predict, lambda_g 5,
lambda_sigma 50 as in evaluation/02_tracking/run_eval.sh). Validation windows: the last 20% of each recording, separated from the
training columns by one window length (Hankel columns overlap), evaluated on the rms position error over the horizon.
usage (repo root, the sim venv): python evaluation/01_prediction/prediction_error_eval.py [n windows per surface (0 = all)]
-> evaluation/01_prediction/prediction_results.json, then visualize_prediction_errors.py.
Other data: PRED_DATA (data folder), PRED_LABELS (dataset label:name of the two regimes, e.g. "0:mu_03,2:mu_10" for the random
excitation data in data/dpc_experiments/01_prediction on branch archive/full-2026-10, whose labels are 0 / 1 / 2 = mu 0.3 / 0.7 / 1.0), PRED_OUT (result file),
PRED_KEYS (comma-separated subset of the settings), PRED_TINI (initialization horizon, default 5), PRED_CACHE=0 to merge the data afresh instead of reusing the cached merge."""
import json, os, sys, time
from copy import copy
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "external/dpc"))
from hydra import compose, initialize_config_dir
import external.dpc.experiment as experiment
from external.dpc.dataset.trajectory_dataset import TrajectoryDataset

N_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LAMBDA_G, LAMBDA_SIGMA, NUM_TRAJ, VAL_SHARE, SEED = 5.0, 50.0, 300, 0.2, 0
LABELS = {int(k): v for k, v in (x.split(":") for x in os.environ.get("PRED_LABELS", "0:mu_05,1:mu_11").split(","))}
DATA = os.environ.get("PRED_DATA", os.path.join(os.getcwd(), "data/dpc_experiments/02_tracking_v2"))

with initialize_config_dir(config_dir=os.path.join(os.path.dirname(experiment.__file__), "config"), version_base=None):
    cfg = compose(config_name="config", overrides=[
        "+experiment=vehicle_dpc_eval_multisurface", f"dataset.path={DATA}",
        f"++dataset.reuse_cached_merge={'true' if os.environ.get('PRED_CACHE', '1') == '1' else 'false'}", f"controller.lambda_g={LAMBDA_G}", f"controller.lambda_sigma={LAMBDA_SIGMA}",
        f"controller.num_traj={NUM_TRAJ}", f"controller.initialization_horizon={os.environ.get('PRED_TINI', '5')}", "controller.path_sampling.sampling_debug=false", "controller.path_sampling.visualize_once=false"])
data = TrajectoryDataset(cfg)
ctrl = cfg.controller; nu, ny, Ti, N = ctrl.num_inputs, ctrl.num_outputs, ctrl.initialization_horizon, ctrl.prediction_horizon
U, Y, lab = data.input_matrix, data.output_matrix, np.asarray(data.dataset_labels)
sd = np.asarray(data.normalization_stats["output_std"], float)
print(f"columns {U.shape[1]} (labels {dict(zip(*np.unique(lab, return_counts=True)))}), T_ini {Ti}, N {N}")


def predict(Up, Uf, Yp, Yf, u_ini, y_ini, u_f):
    """min lg |g|^2 + ls |Yp g - y_ini|^2 s.t. Up g = u_ini, Uf g = u_f, sum g = 1 (helpers/implied_friction.ls_predict), solved
    in the row space (Woodbury) so that the full datasets (thousands of columns) are cheap."""
    n = Up.shape[1]
    A = np.vstack([Up, Uf, np.ones((1, n))]); b = np.concatenate([u_ini, u_f, [1.0]])
    # H = lg I + ls Yp^T Yp;  H^-1 = (I - Yp^T (lg/ls I + Yp Yp^T)^-1 Yp) / lg
    S = np.linalg.inv(LAMBDA_G / LAMBDA_SIGMA * np.eye(Yp.shape[0]) + Yp @ Yp.T)
    Hinv = lambda X: (X - Yp.T @ (S @ (Yp @ X))) / LAMBDA_G
    f = LAMBDA_SIGMA * (Yp.T @ y_ini)
    HiA, Hif = Hinv(A.T), Hinv(f)
    nu_ = np.linalg.lstsq(A @ HiA, A @ Hif - b, rcond=None)[0]       # KKT: 2H g + A^T nu = 2f, A g = b
    g = Hif - HiA @ nu_
    return Yf @ g


def pos_rms_m(err):            # normalised, aligned frame -> metres
    E = err.reshape(-1, ny).T
    return float(np.sqrt(np.mean((E[0] * sd[0]) ** 2 + (E[1] * sd[1]) ** 2)))


# check the fast solver against the reference implementation once
from helpers.implied_friction import ls_predict
rng = np.random.default_rng(SEED)
cols = rng.choice(U.shape[1], 300, replace=False); j = int(rng.integers(U.shape[1]))
ref = ls_predict(U[:Ti * nu, cols], U[Ti * nu:, cols], Y[:Ti * ny, cols], Y[Ti * ny:, cols], U[:Ti * nu, j], Y[:Ti * ny, j], U[Ti * nu:, j], LAMBDA_G, LAMBDA_SIGMA)
fast = predict(U[:Ti * nu, cols], U[Ti * nu:, cols], Y[:Ti * ny, cols], Y[Ti * ny:, cols], U[:Ti * nu, j], Y[:Ti * ny, j], U[Ti * nu:, j])
print("solver check, max |diff|:", float(np.max(np.abs(ref - fast))))

# train / validation split per recording: last VAL_SHARE of the columns, one window length of gap
train_idx, val_idx = {}, {}
for l in LABELS:
    idx = np.where(lab == l)[0]                      # columns are in recording order
    n_val = int(round(VAL_SHARE * len(idx))); gap = Ti + N
    train_idx[l] = idx[: len(idx) - n_val - gap]; val_idx[l] = idx[len(idx) - n_val:]
    if N_WIN: val_idx[l] = val_idx[l][np.linspace(0, len(val_idx[l]) - 1, N_WIN).astype(int)]
    print(f"{LABELS[l]}: {len(train_idx[l])} training, {len(val_idx[l])} validation columns")
tr_all = np.concatenate([train_idx[l] for l in LABELS])
blocks = lambda c: (U[:Ti * nu, c], U[Ti * nu:, c], Y[:Ti * ny, c], Y[Ti * ny:, c])

# a dataset view with the pooled training columns only, for the samplers
pool = copy(data); pool.input_matrix, pool.output_matrix, pool.dataset_labels = U[:, tr_all], Y[:, tr_all], lab[tr_all]
pool.oracle_label = None


def sampled(method, u_ini, y_ini, u_f, only_label=None):
    cfg_s = copy(cfg); cfg_s.controller.path_sampling.sampling = method
    pool.oracle_label = only_label                   # None: the whole pool; a label: that dataset only (as the closed-loop oracle)
    Up, Uf, Yp, Yf, _ = pool.select_trajectories(cfg_s, initial_inputs=u_ini, initial_outputs=y_ini, reference_inputs=u_f)
    pool.oracle_label = None
    return Up, Uf, Yp, Yf


KEYS = ("matched_contextual", "mismatched_contextual", "pooled_contextual", "pooled_random", "matched_full", "mismatched_full", "pooled_full")
if os.environ.get("PRED_KEYS"):
    KEYS = tuple(os.environ["PRED_KEYS"].split(","))
results = {k: {} for k in KEYS}
t0 = time.time()
for l, name in LABELS.items():
    other = next(o for o in LABELS if o != l)        # the other regime of the comparison
    e = {k: [] for k in results}
    for c in val_idx[l]:
        u_ini, y_ini, u_f, y_f = U[:Ti * nu, c], Y[:Ti * ny, c], U[Ti * nu:, c], Y[Ti * ny:, c]
        run = {
            "matched_contextual": lambda: predict(*sampled("contextual", u_ini, y_ini, u_f, l), u_ini, y_ini, u_f),
            "mismatched_contextual": lambda: predict(*sampled("contextual", u_ini, y_ini, u_f, other), u_ini, y_ini, u_f),
            "pooled_contextual": lambda: predict(*sampled("contextual", u_ini, y_ini, u_f), u_ini, y_ini, u_f),
            "pooled_random": lambda: predict(*sampled("random", u_ini, y_ini, u_f), u_ini, y_ini, u_f),
            "matched_full": lambda: predict(*blocks(train_idx[l]), u_ini, y_ini, u_f),
            "mismatched_full": lambda: predict(*blocks(train_idx[other]), u_ini, y_ini, u_f),
            "pooled_full": lambda: predict(*blocks(tr_all), u_ini, y_ini, u_f)}
        for k in KEYS:
            e[k].append(pos_rms_m(run[k]() - y_f))
    for k in KEYS:
        results[k][name] = e[k]
    print(f"{name} (median cm): " + ", ".join(f"{k} {100 * np.median(e[k]):.1f}" for k in KEYS) + f"  ({time.time() - t0:.0f} s)")
results["config"] = dict(data=os.path.relpath(DATA, os.getcwd()), labels=LABELS, lambda_g=LAMBDA_G, lambda_sigma=LAMBDA_SIGMA, num_traj=NUM_TRAJ,
                         val_share=VAL_SHARE, n_val={LABELS[l]: int(len(val_idx[l])) for l in LABELS}, unit="m (rms position error over the horizon)",
                         t_ini=int(Ti), horizon=int(N), seed=SEED)
out = os.environ.get("PRED_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_results.json"))
json.dump(results, open(out, "w"), indent=1); print("written", out)

"""Offline tuning of segment sampling: replay the controller's selection and predictor on recorded windows.

No simulator. For a recorded window (u_ini, y_ini) with its recorded future inputs u_f, every setting (G, k) runs
  1. segment sampling exactly as online (samplers.segment_precompute / the same residual ranking and greedy
     disjoint selection), and
  2. the controller's predictor exactly as in the QP, with the future inputs fixed to the recorded ones:
        min_g  lambda_g ||g||^2 + lambda_sigma ||Y_p g - y_ini||^2
        s.t.   U_p g = u_ini,  U_f g = u_f,  1^T g = 1
     and the prediction Y_f g is compared to the recorded future outputs.
The query's time neighbourhood is removed from both recordings (they follow the same reference in sync), so the
controller has to work from other laps. Labels are used only for scoring.

Reported per setting, in corners (lateral acceleration at the current step in the top 20%):
  pred     RMS prediction error over the horizon (normalized units)
  sel      share of the selected columns from the matching friction
  use      share of the controller's usage (as in DPC: ||Y g_label||^2) from the matching friction
and the prediction error over all windows.

Usage:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python evaluation/exploratory/segment_sampling/offline/offline_tuning.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "dpc"))
from dataset.data_sampling.samplers import segment_precompute  # noqa: E402

DATASET = "data/dpc_experiments/02_tracking/pure_pursuit_0_5/preprocessed/preprocessed_2026-03-23T15-52-48.npy"
RAW = {0.5: "data/dpc_experiments/02_tracking/pure_pursuit_0_5/inputs_outputs_0.5.npy",
       1.1: "data/dpc_experiments/02_tracking/pure_pursuit_1_1/inputs_outputs_1.1.npy"}
FRICTIONS = np.array([0.5, 1.1])  # dataset labels 0 and 1
T_INI, N, NU, NY = 5, 10, 2, 4
LAMBDA_G, LAMBDA_SIGMA, RIDGE = 1.1, 10.1, 0.1
G_VALUES = [10, 15, 20, 25, 30, 40, 50, 60]
K_VALUES = [2, 3, 4, 6, 8]
MAX_COLUMNS = 400
EXCLUDE = 30  # steps around the query, in both recordings
N_QUERY = 300  # per recording


def predict(Up, Yp, Uf, Yf, u_ini, y_ini, u_f):
    """The controller's predictor with the future inputs fixed (equality-constrained ridge, solved by KKT)."""
    n = Up.shape[1]
    Q = LAMBDA_G * np.eye(n) + LAMBDA_SIGMA * Yp.T @ Yp
    C = np.vstack([Up, Uf, np.ones((1, n))])
    K = np.block([[2 * Q, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
    rhs = np.concatenate([2 * LAMBDA_SIGMA * Yp.T @ y_ini, u_ini, u_f, [1.0]])
    g = np.linalg.lstsq(K, rhs, rcond=None)[0][:n]
    return Yf @ g, g


def main():
    d = np.load(os.path.join(REPO_ROOT, DATASET), allow_pickle=True).item()
    U, Y, labels = np.asarray(d["input_matrix"]), np.asarray(d["output_matrix"]), np.asarray(d["dataset_labels"])
    Up, Uf, Yp, Yf = U[:T_INI * NU], U[T_INI * NU:], Y[:T_INI * NY], Y[T_INI * NY:]
    run_start = {lab: np.flatnonzero(labels == lab)[0] for lab in np.unique(labels)}
    index_in_run = np.arange(len(labels)) - np.array([run_start[l] for l in labels])

    # excitation at the current step of each column: |v * yaw rate| from the raw recordings
    lat = np.empty(len(labels))
    for lab, mu in enumerate(FRICTIONS):
        a = np.load(os.path.join(REPO_ROOT, RAW[mu]))
        la = np.abs(a[:, 6] * np.gradient(np.unwrap(a[:, 7]), 0.1))
        cols = np.flatnonzero(labels == lab)
        lat[cols] = la[index_in_run[cols] + T_INI - 1]
    corner = lat > np.percentile(lat, 80)

    rng = np.random.default_rng(0)
    queries = np.concatenate([rng.choice(np.flatnonzero(labels == lab)[EXCLUDE:-EXCLUDE], N_QUERY, replace=False)
                              for lab in np.unique(labels)])

    print(f"{N_QUERY} recorded windows per friction, {corner[queries].sum()} in corners; "
          f"lambda_g={LAMBDA_G}, lambda_sigma={LAMBDA_SIGMA}\n")
    print(f"{'G':>3} {'k':>2} {'cols':>4} | {'corners: pred':>13} {'sel':>5} {'use':>5} | {'all: pred':>9}")
    results = {}
    for G in G_VALUES:
        cache = segment_precompute(Up, Yp, NU, segment_length=G, ridge=RIDGE)
        starts = cache["starts"]
        rows = {k: [] for k in K_VALUES if k * G <= MAX_COLUMNS}
        for q in queries:
            w = np.concatenate([Up[:, q], Yp[:, q]])
            excluded = np.abs(index_in_run - index_in_run[q]) <= EXCLUDE
            blocked = np.concatenate([[0], np.cumsum(excluded)])
            valid = blocked[starts + G] == blocked[starts]  # segment contains no excluded column
            residuals = np.linalg.norm(cache["M"][valid] @ w + cache["b"][valid], axis=1)
            order = starts[valid][np.argsort(residuals)]
            for k in rows:
                used = np.zeros(len(labels), bool)
                sel = []
                for j in order:
                    if not used[j:j + G].any():
                        used[j:j + G] = True
                        sel.append(np.arange(j, j + G))
                        if len(sel) == k:
                            break
                sel = np.concatenate(sel)
                pred, g = predict(Up[:, sel], Yp[:, sel], Uf[:, sel], Yf[:, sel], Up[:, q], Yp[:, q], Uf[:, q])
                err = np.sqrt(np.mean((pred - Yf[:, q]) ** 2))
                match = labels[sel] == labels[q]
                usage = [sum(np.linalg.norm(M[:, sel][:, m] @ g[m]) ** 2 for M in (Yp, Yf)) for m in (match, ~match)]
                rows[k].append((err, match.mean(), usage[0] / (sum(usage) + 1e-12), corner[q]))
        for k, r in rows.items():
            r = np.array(r)
            c = r[:, 3].astype(bool)
            results[(G, k)] = (np.median(r[c, 0]), r[c, 1].mean(), r[c, 2].mean(), np.median(r[:, 0]))
            pc, sc, uc, pa = results[(G, k)]
            print(f"{G:>3} {k:>2} {G * k:>4} | {pc:>13.3f} {sc:>5.0%} {uc:>5.0%} | {pa:>9.3f}")
        print()


if __name__ == "__main__":
    main()

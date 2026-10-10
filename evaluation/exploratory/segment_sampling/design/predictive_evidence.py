"""Can the recent past tell the friction apart, if each dataset has to PREDICT it?

Offline test of a candidate replacement for the evidence weighting in the sampler.
At a time t, for each friction dataset k:
  - initialization window: the T_INI steps before t-M
  - test: the M steps from t-M to t, predicted by dataset k's DPC predictor from the
    initialization window and the inputs that were actually applied
  - e_k: RMS prediction error over the test (per-channel scaled)
The dataset that predicts better is the evidence. M is the only memory; M = 0 has no
test and stands for the current mechanism (fit of the current window only).

The two recordings follow the same reference in sync, so the query's time neighbourhood
is removed from BOTH datasets: each must predict from other laps, as the controller would.

Reported, per scope (by lateral acceleration at t, i.e. when the decision matters):
  acc     fraction of query points where the matching dataset has the smaller error
  ratio   median e_wrong / e_matching
  p       mean posterior of the matching dataset, p ~ exp(-M e^2 / (2 sigma^2)), with
          sigma the matching error measured on a separate calibration half

Usage:
    .venv/bin/python evaluation/exploratory/segment_sampling/design/predictive_evidence.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "observability"))
from cross_prediction_channels import DATA, T_INI, N_CANDIDATES, load_channels, predict, lateral_acceleration  # noqa: E402

NAMES = ["x", "y", "v", "yaw"]
M_VALUES = [0, 5, 10, 20]
N_QUERY = 400  # query points per friction (half calibration, half test)
FIT_NEIGHBOURS, FIT_RIDGE = 20, 0.1  # current evidence_weights settings, for M = 0


def windows(ch, total):
    """Columns of length `total` in the local frame at the end of the initialization part."""
    u = np.stack([ch["accel"], ch["steering"]], axis=1)
    y = np.stack([ch[k] for k in NAMES], axis=1)
    U, Y = [], []
    for s in range(len(ch["x"]) - total):
        yw = y[s:s + total].copy()
        ref, psi = yw[T_INI - 1, :2].copy(), ch["yaw"][s + T_INI - 1]
        c, si = np.cos(-psi), np.sin(-psi)
        d = yw[:, :2] - ref
        yw[:, 0], yw[:, 1] = c * d[:, 0] - si * d[:, 1], si * d[:, 0] + c * d[:, 1]
        yw[:, 3] -= ch["yaw"][s + T_INI - 1]
        U.append(u[s:s + total].ravel())
        Y.append(yw.ravel())
    return np.array(U).T, np.array(Y).T


def fit_error(Up, Yp, u_ini, y_ini, scale):
    """Current mechanism: affine ridge fit of the window by the nearest columns."""
    A = np.vstack([Up, Yp / scale[:, None]])
    w = np.concatenate([u_ini, y_ini / scale])
    m = A.shape[1]
    K = np.block([[A.T @ A + FIT_RIDGE * np.eye(m), np.ones((m, 1))], [np.ones((1, m)), np.zeros((1, 1))]])
    g = np.linalg.solve(K, np.r_[A.T @ w, 1.0])[:m]
    return np.sqrt(np.mean((A @ g - w) ** 2))


def errors(M, rng):
    """e_k for every query point: {mu_true: (queries, e_matching, e_wrong, lat_at_t)}."""
    total = T_INI + max(M, 1)
    split_u, split_y = T_INI * 2, T_INI * len(NAMES)
    chans = {mu: load_channels(p) for mu, p in DATA.items()}
    data = {mu: windows(ch, total) for mu, ch in chans.items()}
    scale = np.abs(np.hstack([data[mu][1] for mu in data])).mean(axis=1).reshape(-1, len(NAMES)).mean(axis=0)
    scale_p, scale_f = np.tile(scale, T_INI), np.tile(scale, max(M, 1))

    out = {}
    for mu in DATA:
        other = [m for m in DATA if m != mu][0]
        U_e, Y_e = data[mu]
        lat = lateral_acceleration(chans[mu])
        # the same decision times t for every M (fixed rng); the window ending at t starts at s
        t = rng.choice(np.arange(T_INI + max(M_VALUES), U_e.shape[1]), N_QUERY, replace=False)
        q = t - T_INI - M
        e = {mu: [], other: []}
        for s in q:
            u_ini, y_ini = U_e[:split_u, s], Y_e[:split_y, s]
            u_f, y_true = U_e[split_u:, s], Y_e[split_y:, s]
            for k in (mu, other):
                U_d, Y_d = data[k]
                cols = np.arange(U_d.shape[1])
                cols = cols[np.abs(cols - s) > total + 10]  # query's neighbourhood, in both runs
                d = np.linalg.norm(Y_d[:split_y, cols] - y_ini[:, None], axis=0)
                if M == 0:
                    sel = cols[np.argsort(d)[:FIT_NEIGHBOURS]]
                    e[k].append(fit_error(U_d[:split_u, sel], Y_d[:split_y, sel], u_ini, y_ini, scale_p))
                else:
                    sel = cols[np.argsort(d)[:N_CANDIDATES]]
                    pred = predict(U_d[:split_u, sel], Y_d[:split_y, sel], U_d[split_u:, sel], Y_d[split_y:, sel],
                                   u_ini, y_ini, u_f, scale_f)
                    e[k].append(np.sqrt(np.mean((pred - y_true / scale_f) ** 2)))
        out[mu] = (np.array(e[mu]), np.array(e[other]), lat[t])
    return out


def main():
    lat_all = np.concatenate([lateral_acceleration(load_channels(p)) for p in DATA.values()])
    hi, lo = np.percentile(lat_all, 80), np.percentile(lat_all, 50)
    scopes = {"all": lambda a: np.ones_like(a, bool), f"corners (a_lat>{hi:.1f})": lambda a: a > hi,
              f"straights (a_lat<{lo:.1f})": lambda a: a < lo}
    print(f"t_ini={T_INI}, {N_CANDIDATES} candidates, {N_QUERY} query points per friction; "
          f"M=0 is the current window fit ({FIT_NEIGHBOURS} neighbours, ridge {FIT_RIDGE})\n")
    print(f"{'M':>3} {'scope':<22} " + "  ".join(f"{'mu=' + str(mu) + ' acc / ratio / p':>26}" for mu in DATA))
    for M in M_VALUES:
        res = errors(M, np.random.default_rng(0))
        for name, fn in scopes.items():
            cells = []
            for mu, (e_match, e_wrong, lat) in res.items():
                calib = np.arange(len(e_match)) % 2 == 0
                sigma = np.sqrt(np.mean(e_match[calib] ** 2))
                m = fn(lat) & ~calib
                n_eff = max(M, 1)
                logit = n_eff * (e_wrong[m] ** 2 - e_match[m] ** 2) / (2 * sigma ** 2)
                p = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
                cells.append(f"{np.mean(e_match[m] < e_wrong[m]):>6.0%} / {np.median(e_wrong[m] / e_match[m]):>5.2f}"
                             f" / {p.mean():>5.2f}  (n={m.sum():>3})")
            print(f"{M:>3} {name:<22} " + "  ".join(f"{c:>26}" for c in cells))
        print()
    print("acc and ratio do not depend on sigma; p is the posterior the sampler would use.\n"
          "After a friction change the evidence is fully replaced after M steps.")


if __name__ == "__main__":
    main()

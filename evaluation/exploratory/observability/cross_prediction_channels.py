"""Does adding yaw rate to the output vector make the friction regimes easier to tell apart?

Offline test, no controller involved. For each output-channel variant we measure the
distinguishability margin

    eps = (prediction error using the WRONG friction subset)
        - (prediction error using the MATCHING friction subset)

A larger eps means the initialization window carries more information about the
active friction condition, which is the quantity the implicit selection relies on.

Usage:
    .venv/bin/python evaluation/exploratory/observability/cross_prediction_channels.py
"""
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = {
    0.5: "data/dpc_experiments/02_tracking/pure_pursuit_0_5/inputs_outputs_0.5.npy",
    1.1: "data/dpc_experiments/02_tracking/pure_pursuit_1_1/inputs_outputs_1.1.npy",
}
DT = 0.1
T_INI, HORIZON = 5, 20
N_CANDIDATES = 100        # columns offered to the regression, as in the controller
LAMBDA_G, LAMBDA_SIGMA = 1.1, 10.1

# Channel variants: which output signals the window and the prediction use.
# x, y and yaw are in the trajectory-local frame (see align()).
VARIANTS = {
    "base (x, y, v, yaw)": ["x", "y", "v", "yaw"],
    "+ yaw rate": ["x", "y", "v", "yaw", "yaw_rate"],
    "+ a_lat (v * yaw rate)": ["x", "y", "v", "yaw", "a_lat"],
    "+ a_lat + yaw rate": ["x", "y", "v", "yaw", "yaw_rate", "a_lat"],
}


def load_channels(path):
    a = np.load(os.path.join(REPO_ROOT, path))
    yaw = np.unwrap(a[:, 7])
    return {
        "accel": a[:, 2], "steering": a[:, 3],
        "x": a[:, 4], "y": a[:, 5], "v": a[:, 6], "yaw": yaw,
        "yaw_rate": np.gradient(yaw, DT),
        # v * psi_dot: DPC is linear in g, so a product of channels is information
        # the matching cannot construct from the individual signals.
        "a_lat": a[:, 6] * np.gradient(yaw, DT),
    }


def windows(ch, names):
    """Stacked past/future matrices, each column one trajectory in its local frame."""
    total = T_INI + HORIZON
    n = len(ch["x"]) - total
    u = np.stack([ch["accel"], ch["steering"]], axis=1)          # (T, 2)
    y = np.stack([ch[k] for k in names], axis=1)                 # (T, n_y)
    ix, iy = (names.index("x"), names.index("y")) if "x" in names else (None, None)
    iyaw = names.index("yaw") if "yaw" in names else None

    U, Y = [], []
    for s in range(0, n, 2):
        yw = y[s:s + total].copy()
        if ix is not None:
            # same alignment as the controller: origin and heading at t_ini-1
            ref, psi = yw[T_INI - 1, [ix, iy]].copy(), ch["yaw"][s + T_INI - 1]
            c, si = np.cos(-psi), np.sin(-psi)
            d = yw[:, [ix, iy]] - ref
            yw[:, ix], yw[:, iy] = c * d[:, 0] - si * d[:, 1], si * d[:, 0] + c * d[:, 1]
        if iyaw is not None:
            yw[:, iyaw] -= ch["yaw"][s + T_INI - 1]
        U.append(u[s:s + total].ravel())
        Y.append(yw.ravel())
    return np.array(U).T, np.array(Y).T          # (total*2, N), (total*n_y, N)


def predict(Up, Yp, Uf, Yf, u_ini, y_ini, u_f, scale):
    """Regularized DPC prediction, solved in closed form (ridge)."""
    A = np.vstack([Up * LAMBDA_SIGMA, Yp * LAMBDA_SIGMA, Uf * LAMBDA_SIGMA])
    b = np.concatenate([u_ini * LAMBDA_SIGMA, y_ini * LAMBDA_SIGMA, u_f * LAMBDA_SIGMA])
    g = np.linalg.solve(A.T @ A + LAMBDA_G * np.eye(A.shape[1]), A.T @ b)
    return (Yf @ g) / scale


def lateral_acceleration(ch):
    """|v * yaw_rate| per sample, the excitation level friction acts on."""
    return np.abs(ch["v"] * ch["yaw_rate"])


def evaluate(names, n_eval=150, seed=0, corners_only=False):
    rng = np.random.default_rng(seed)
    n_y = len(names)
    split_p, split_u = T_INI * n_y, T_INI * 2
    data, excitation = {}, {}
    for mu, path in DATA.items():
        ch = load_channels(path)
        U, Y = windows(ch, names)
        data[mu] = (U, Y)
        lat = lateral_acceleration(ch)
        # excitation of each window, taken over its initialization part
        excitation[mu] = np.array([lat[s:s + T_INI].max()
                                   for s in range(0, len(ch["x"]) - T_INI - HORIZON, 2)])
    # per-channel scale, so no single signal dominates the error metric
    scale = np.abs(np.hstack([data[mu][1] for mu in data])).mean(axis=1).reshape(-1, n_y).mean(axis=0)
    scale = np.tile(scale, HORIZON)

    out = {}
    for mu_eval in DATA:
        U_e, Y_e = data[mu_eval]
        pool = np.arange(U_e.shape[1])
        if corners_only:
            lat = excitation[mu_eval][:U_e.shape[1]]
            pool = pool[lat > np.percentile(lat, 80)]
        idx = rng.choice(pool, min(n_eval, len(pool)), replace=False)
        for mu_data in DATA:
            U_d, Y_d = data[mu_data]
            errs = []
            for i in idx:
                u_ini, u_f = U_e[:split_u, i], U_e[split_u:, i]
                y_ini, y_true = Y_e[:split_p, i], Y_e[split_p:, i]
                # Both runs follow the same reference on the same track, so the
                # window at the same index is a near-duplicate in EITHER subset.
                # Exclude that neighbourhood in both cases, or the matched subset
                # is handicapped against its own twin in the other subset.
                cols = np.arange(U_d.shape[1])
                cols = cols[np.abs(cols - i) > HORIZON]
                # retrieval: nearest windows in the past signals, as online
                d = np.linalg.norm(Y_d[:split_p, cols] - y_ini[:, None], axis=0)
                sel = cols[np.argsort(d)[:N_CANDIDATES]]
                pred = predict(U_d[:split_u, sel], Y_d[:split_p, sel],
                               U_d[split_u:, sel], Y_d[split_p:, sel],
                               u_ini, y_ini, u_f, scale)
                errs.append(np.sqrt(np.mean((pred - y_true / scale) ** 2)))
            out[(mu_eval, mu_data)] = float(np.median(errs))
    return out


def main():
    print(f"t_ini={T_INI}, horizon={HORIZON}, {N_CANDIDATES} candidate columns, "
          f"lambda_g={LAMBDA_G}, lambda_sigma={LAMBDA_SIGMA}\n")
    print(f"{'variant':<24} {'eval mu':>8} {'matched':>9} {'mismatched':>11} {'eps':>8} {'ratio':>7}")
    print("-" * 72)
    for label, names in VARIANTS.items():
      for scope, corners in [("all", False), ("corners", True)]:
        res = evaluate(names, corners_only=corners)
        for mu in DATA:
            other = [m for m in DATA if m != mu][0]
            matched, mismatched = res[(mu, mu)], res[(mu, other)]
            print(f"{label + ' [' + scope + ']':<24} {mu:>8} {matched:>9.4f} {mismatched:>11.4f} "
                  f"{mismatched - matched:>+8.4f} {mismatched / matched:>7.2f}")
    print("\neps > 0 means the wrong-friction data predicts worse, i.e. the window is\n"
          "informative about friction. Compare eps across variants, not across mu.")


if __name__ == "__main__":
    main()

"""Which subset influences the prediction, measured offline on a pooled dataset.

This is the offline analogue of the paper's implied-friction figure: predict a window
from the POOLED dataset and attribute the predicted future to the friction subsets via

    eta_j = ||Y_f,j g_j|| / sum_l ||Y_f,l g_l||,   mu_impl = sum_j eta_j mu_j

Includes two controls that the closed-loop version cannot easily provide:

  placebo   - the two subsets carry the SAME friction (one run split in half).
              Any separation here is an artifact of the metric, not friction.
  shuffled  - real subsets, labels shuffled across columns. Same expectation.

Usage:
    .venv/bin/python evaluation/exploratory/observability/pooled_influence.py
"""
import os

import numpy as np

from cross_prediction_channels import (DATA, HORIZON, LAMBDA_G, LAMBDA_SIGMA, N_CANDIDATES,
                                       T_INI, lateral_acceleration, load_channels, windows)

CHANNELS = ["x", "y", "v", "yaw"]
FRICTIONS = np.array(sorted(DATA))


def solve(Up, Yp, Uf, u_ini, y_ini, u_f):
    A = np.vstack([Up, Yp, Uf]) * LAMBDA_SIGMA
    b = np.concatenate([u_ini, y_ini, u_f]) * LAMBDA_SIGMA
    return np.linalg.solve(A.T @ A + LAMBDA_G * np.eye(A.shape[1]), A.T @ b)


def influence(Yf, g, labels, groups):
    contributions = np.array([np.linalg.norm(Yf[:, labels == j] @ g[labels == j])
                              for j in groups])
    total = contributions.sum()
    return contributions / total if total > 0 else np.full(len(groups), np.nan)


def run(mode, n_eval=200, seed=0, exclude_twins=True):
    """mode: 'real' | 'placebo' | 'shuffled'.

    exclude_twins=False keeps the same-track-position trajectories of every subset in
    the candidate pool, which is the situation the closed-loop controller is actually in:
    it drives the track the data was recorded on, so a matched pair of trajectories
    (one per friction) exists for the current position.
    """
    rng = np.random.default_rng(seed)
    n_y = len(CHANNELS)
    split_p, split_u = T_INI * n_y, T_INI * 2

    mats, labels, exc = [], [], {}
    for k, mu in enumerate(FRICTIONS):
        ch = load_channels(DATA[mu])
        U, Y = windows(ch, CHANNELS)
        lat = lateral_acceleration(ch)
        exc[mu] = np.array([lat[s:s + T_INI].max()
                            for s in range(0, len(ch["x"]) - T_INI - HORIZON, 2)])[:U.shape[1]]
        mats.append((U, Y))
        labels.append(np.full(U.shape[1], k))

    if mode == "placebo":
        # both "subsets" are the same friction: first run split into halves
        U, Y = mats[0]
        half = U.shape[1] // 2
        mats = [(U[:, :half], Y[:, :half]), (U[:, half:], Y[:, half:])]
        labels = [np.zeros(half, int), np.ones(U.shape[1] - half, int)]

    U_all = np.hstack([m[0] for m in mats])
    Y_all = np.hstack([m[1] for m in mats])
    lab_all = np.concatenate(labels)
    if mode == "shuffled":
        lab_all = rng.permutation(lab_all)

    groups = np.unique(lab_all)
    results = {}
    for k, mu in enumerate(FRICTIONS):
        U_e, Y_e = mats[k] if mode != "placebo" else mats[0]
        pool = np.arange(U_e.shape[1])
        lat = exc[mu][:U_e.shape[1]] if mode != "placebo" else exc[FRICTIONS[0]][:U_e.shape[1]]
        corner_pool = pool[lat > np.percentile(lat, 80)]
        for scope, candidates in [("all", pool), ("corners", corner_pool)]:
            idx = rng.choice(candidates, min(n_eval, len(candidates)), replace=False)
            implied = []
            for i in idx:
                u_ini, u_f = U_e[:split_u, i], U_e[split_u:, i]
                y_ini = Y_e[:split_p, i]
                # Retrieval over the pooled data. Both runs follow the same reference,
                # so the window at the same index is a near-duplicate in EVERY subset:
                # exclude that neighbourhood per subset, or the twin in the other
                # subset wins the retrieval and the attribution flips mechanically.
                d = np.linalg.norm(Y_all[:split_p] - y_ini[:, None], axis=0)
                for j in groups if exclude_twins else []:
                    in_subset = np.flatnonzero(lab_all == j)
                    local = np.arange(len(in_subset))
                    d[in_subset[np.abs(local - i) <= HORIZON]] = np.inf
                sel = np.argsort(d)[:N_CANDIDATES]
                g = solve(U_all[:split_u, sel], Y_all[:split_p, sel], U_all[split_u:, sel],
                          u_ini, y_ini, u_f)
                eta = influence(Y_all[split_p:, sel], g, lab_all[sel], groups)
                implied.append(float(eta @ FRICTIONS))
            results[(mu, scope)] = np.array(implied)
        if mode == "placebo":
            break
    return results


def main():
    neutral = FRICTIONS.mean()
    for mode, kwargs in [("real", {}), ("real, twins kept", {"exclude_twins": False}),
                         ("placebo", {}), ("shuffled", {}),
                         ("shuffled, twins kept", {"exclude_twins": False})]:
        res = run(mode.split(",")[0], **kwargs)
        print(f"\n=== {mode} ===")
        print(f"{'eval mu':>8} {'scope':>9} {'implied (median)':>18} {'share of candidates':>21}")
        for (mu, scope), vals in res.items():
            print(f"{mu:>8} {scope:>9} {np.nanmedian(vals):>18.3f}")
        keys = [k for k in res if k[1] == "corners"]
        if len(keys) == 2:
            sep = np.nanmedian(res[keys[1]]) - np.nanmedian(res[keys[0]])
            span = FRICTIONS[-1] - FRICTIONS[0]
            print(f"  corner separation: {sep:+.3f} of {span:+.2f} ({100 * sep / span:.0f}%)")
    print(f"\nneutral value (no discrimination) = {neutral:.2f}. Placebo and shuffled should "
          f"sit there;\nany separation they show is a property of the metric, not of friction.")


if __name__ == "__main__":
    main()

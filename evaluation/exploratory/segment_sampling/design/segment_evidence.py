"""Label-free window-fit evidence: short recording segments as the groups.

The window fit of predictive_evidence.py (M = 0) tells the friction apart in corners,
but it needs groups of columns recorded under the same conditions. Labels assume a
whole recording is one condition. Here the groups are contiguous segments of G steps
from either recording, so conditions only need to be constant within one segment.

For each query window and each segment: the (up to) FIT_NEIGHBOURS closest columns of
that segment fit the window (affine ridge fit, as in the sampler); r_j is the residual.
G = whole recording reproduces the labelled M = 0 result.

Reported, per scope (lateral acceleration at the current step):
  best    the best-fitting segment comes from the matching friction
  top5    share of the 5 best-fitting segments from the matching friction
Labels are used only for this scoring.

Usage:
    .venv/bin/python evaluation/exploratory/segment_sampling/design/segment_evidence.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "observability"))
from cross_prediction_channels import DATA, T_INI, load_channels, lateral_acceleration  # noqa: E402
from predictive_evidence import NAMES, M_VALUES, N_QUERY, FIT_NEIGHBOURS, windows, fit_error  # noqa: E402

G_VALUES = [25, 50, 100, 250, 500, None]  # None = whole recording (= labels); one lap is ~490 steps


def main():
    total = T_INI + 1
    split_u, split_y = T_INI * 2, T_INI * len(NAMES)
    chans = {mu: load_channels(p) for mu, p in DATA.items()}
    data = {mu: windows(ch, total) for mu, ch in chans.items()}
    scale = np.abs(np.hstack([data[mu][1] for mu in data])).mean(axis=1).reshape(-1, len(NAMES)).mean(axis=0)
    scale_p = np.tile(scale, T_INI)
    lat = {mu: lateral_acceleration(ch) for mu, ch in chans.items()}
    lat_all = np.concatenate(list(lat.values()))
    hi, lo = np.percentile(lat_all, 80), np.percentile(lat_all, 50)
    scopes = {"all": lambda a: np.ones_like(a, bool), f"corners (a_lat>{hi:.1f})": lambda a: a > hi,
              f"straights (a_lat<{lo:.1f})": lambda a: a < lo}

    # same query times as predictive_evidence.py (M = 0)
    rng = np.random.default_rng(0)
    queries = {}
    for mu in DATA:
        t = rng.choice(np.arange(T_INI + max(M_VALUES), data[mu][0].shape[1]), N_QUERY, replace=False)
        queries[mu] = t

    print(f"{N_QUERY} query points per friction, {FIT_NEIGHBOURS} neighbours per segment fit; "
          f"chance = 50%\n")
    print(f"{'G':>6} {'segments':>8} {'scope':<22} " + "  ".join(f"{'mu=' + str(mu) + ' best / top5':>22}" for mu in DATA))
    for G in G_VALUES:
        segs = []  # (mu, first column, last column + 1)
        for mu in DATA:
            n = data[mu][0].shape[1]
            step = n if G is None else G
            segs += [(mu, a, min(a + step, n)) for a in range(0, n, step)]
        rows = {name: [] for name in scopes}
        for mu in DATA:
            res = []
            for t in queries[mu]:
                s = t - T_INI
                u_ini, y_ini = data[mu][0][:split_u, s], data[mu][1][:split_y, s]
                dist = {k: np.linalg.norm(data[k][1][:split_y] - y_ini[:, None], axis=0) for k in DATA}
                r, lab = [], []
                for k, a, b in segs:
                    cols = np.arange(a, b)
                    cols = cols[np.abs(cols - s) > total + 10]  # query's neighbourhood, in both runs
                    if len(cols) < 2:
                        continue
                    near = cols[np.argsort(dist[k][cols])[:FIT_NEIGHBOURS]]
                    r.append(fit_error(data[k][0][:split_u, near], data[k][1][:split_y, near], u_ini, y_ini, scale_p))
                    lab.append(k)
                order = np.argsort(r)
                lab = np.array(lab)[order]
                res.append([lab[0] == mu, np.mean(lab[:5] == mu)])
            res = np.array(res, float)
            for name, fn in scopes.items():
                m = fn(lat[mu][queries[mu]])
                b, t5 = res[m].mean(axis=0)
                rows[name].append(f"{b:.0%} / {t5:.0%} (n={m.sum():>3})")
        label = "full" if G is None else str(G)
        for name in scopes:
            print(f"{label:>6} {len(segs):>8} {name:<22} " + "  ".join(f"{c:>22}" for c in rows[name]))
        print()


if __name__ == "__main__":
    main()

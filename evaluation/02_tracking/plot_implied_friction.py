"""Prediction-based implied friction figure (helpers/implied_friction.py logs): implied mu along the lap (laps >= 1, all runs),
only where the mu-0.5-only and mu-1.1-only predictions differ by more than THR (friction readable), gray elsewhere.
usage (sim repo dir): <out.png> <map> <thr m> <log> ..."""
import json, sys, numpy as np, yaml
from argparse import Namespace
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
out, map_name, thr, logs = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
conf = Namespace(**yaml.load(open(f"configs/config_{map_name}.yaml"), Loader=yaml.FullLoader)); wp = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
L = wp[-1, 0]; edges = np.arange(0, L + 4, 4.0); sc = edges[:-1] + 2; nb = len(sc)
vals = [[] for _ in range(nb)]; tfb = [[] for _ in range(nb)]; readable = [[] for _ in range(nb)]
for f in logs:
    d = json.load(open(f)); lap = np.array(d["lap_n"]); p = np.array(d["track_progress"], float); fr = np.array(d["friction"], float)
    n = lambda k: np.array([np.nan if v is None else v for v in d[k]], float); mu, di = n("implied_mu"), n("implied_dist")
    for k in np.where(lap >= 1)[0]:
        b = min(int(p[k] // 4), nb - 1); tfb[b].append(fr[k])
        if np.isfinite(mu[k]): readable[b].append(di[k] > thr)
        if np.isfinite(mu[k]) and di[k] > thr: vals[b].append(mu[k])
im = np.array([np.mean(v) if len(v) >= 3 else np.nan for v in vals]); lo = np.array([np.percentile(v, 25) if len(v) >= 3 else np.nan for v in vals]); hi = np.array([np.percentile(v, 75) if len(v) >= 3 else np.nan for v in vals])
tf = np.array([np.median(v) if v else np.nan for v in tfb]); share = np.array([np.mean(v) if v else 0 for v in readable])
for z in (0.5, 0.8, 1.1):
    m = np.isclose(tf, z); print(f"true mu {z}: implied mean {np.nanmean(im[m]):.2f} over readable sections, readable share {np.mean(share[m])*100:.0f}%")
x = np.interp(sc, wp[:, 0], wp[:, 1]); y = np.interp(sc, wp[:, 0], wp[:, 2]); dx, dy = np.gradient(x), np.gradient(y); nr = np.hypot(dx, dy) + 1e-9; nx, ny = -dy / nr, dx / nr
seg = lambda X, Y: np.concatenate([np.c_[X, Y][:-1, None], np.c_[X, Y][1:, None]], 1); off = 6.0
fig = plt.figure(figsize=(13, 5.2), constrained_layout=True); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
ax = fig.add_subplot(gs[0]); nc = plt.Normalize(0.5, 1.1)
ax.plot(x + off * nx, y + off * ny, color="#cfcfcf", lw=4, solid_capstyle="butt")
for X, Y, val in ((x - off * nx, y - off * ny, tf), (x + off * nx, y + off * ny, im)):
    lc = LineCollection(seg(X, Y), cmap="coolwarm", norm=nc, lw=4); lc.set_array(np.ma.masked_invalid(val[:-1])); ax.add_collection(lc)
ax.set_aspect("equal"); ax.autoscale(); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"inner: true friction | outer: implied friction of the pooled DPC\n(gray: friction not readable, the two datasets predict within {100*thr:.0f} cm)", fontsize=9)
fig.colorbar(lc, ax=ax, shrink=0.7, label="friction / implied friction")
b = fig.add_subplot(gs[1])
b.plot(sc, tf, color="#444444", lw=1.5, label="true friction")
b.fill_between(sc, lo, hi, color="#1b7f3b", alpha=0.18, lw=0, label="implied friction, interquartile range")
b.plot(sc, im, color="#1b7f3b", lw=1.6, label="implied friction (mean, readable steps)")
b.set_ylim(0.45, 1.15); b.set_xlim(0, L); b.set_xlabel("progress on the track [m]"); b.set_ylabel("friction"); b.legend(fontsize=8, loc="upper left")
for s in ("top", "right"): b.spines[s].set_visible(False)
fig.suptitle("Prediction-based implied friction: which friction the pooled DPC's prediction corresponds to (3 runs x 10 laps)", fontsize=10)
fig.savefig(out, dpi=140); print("written", out)

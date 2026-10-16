"""Paper figures from the evaluation logs (evaluation/02_tracking/run_eval.sh):
evaluation/02_tracking/tracking_error_<map>: tracking error along the lap (median over all evaluated laps of all seeds, band:
min-max of DPC pooled and NMPC) on the friction profile; for l_shape a second panel with the prediction error of the DPC
variants (position error of the 10-step prediction vs the positions reached, rms over the horizon).
evaluation/02_tracking/implied_friction_<map>: track map, inner ring true friction, outer ring the prediction-based implied
friction of the pooled DPC (helpers/implied_friction.py), gray where the two single-friction predictions differ by less than
3 cm (friction not readable from the prediction).
usage (repo root): python visualization_dpc.py [l_shape|Nuerburgring] [results dir (evaluation/02_tracking/results)]"""
import glob
import json
import os
import sys
from argparse import Namespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from helpers import paper_style as ps
from helpers.eval_metrics import prediction_error

map_name = sys.argv[1] if len(sys.argv) > 1 else "l_shape"
results = sys.argv[2] if len(sys.argv) > 2 else "evaluation/02_tracking/results"
arms = ("NMPC", "pooled", "oracle", "inverse") if map_name == "l_shape" else ("NMPC", "pooled")
bin_m = 2.0 if map_name == "l_shape" else 10.0
readable_m = 0.03
ps.apply()

with open("configs/config_%s.yaml" % map_name) as file:
    conf = Namespace(**yaml.load(file, Loader=yaml.FullLoader))
waypoints = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
track_len = waypoints[-1, 0]
edges = np.arange(0, track_len + bin_m, bin_m)
centers = edges[:-1] + bin_m / 2
n_bins = len(centers)


def logs(arm):
    pattern = f"log_nmpc_eval_{map_name}_s*" if arm == "NMPC" else f"log_dpc_eval_{map_name}_eval_{arm}_s*"
    return sorted(glob.glob(os.path.join(results, pattern)))


def load_log(name):
    with open(name, "r") as f:
        data = json.load(f)
    nan = lambda v: np.array([np.nan if x is None else x for x in v], dtype=float)
    out = {k: nan(data[k]) for k in ("time", "x", "y", "lap_n", "friction", "tracking_error", "track_progress")}
    for k in ("implied_mu", "implied_dist"):
        if k in data:
            out[k] = nan(data[k])
    if "pred_xy" in data:
        out["prediction_error"] = prediction_error(data["pred_xy"], out["x"], out["y"])
    return out


def binned(runs, key, stat=np.median):
    """per progress bin: median, min and max over all evaluated laps (lap >= 1) of all runs"""
    vals = [[] for _ in range(n_bins)]
    for d in runs:
        valid = (d["lap_n"] >= 1) & np.isfinite(d[key])
        for p, v in zip(d["track_progress"][valid], d[key][valid]):
            vals[min(int(p // bin_m), n_bins - 1)].append(v)
    return (np.array([stat(v) if v else np.nan for v in vals]), np.array([np.min(v) if v else np.nan for v in vals]),
            np.array([np.max(v) if v else np.nan for v in vals]))


data = {arm: [load_log(f) for f in logs(arm)] for arm in arms}
for arm in arms:
    print(f"{arm}: {len(data[arm])} runs from {results}")
friction, _, _ = binned(data["pooled"], "friction")


def friction_background(ax):
    ax.imshow(friction[np.newaxis, :], extent=[0, track_len, 0, 1], aspect="auto", cmap=ps.FRICTION_CMAP, vmin=ps.FRICTION_MIN,
              vmax=ps.FRICTION_MAX, alpha=0.35, origin="lower", interpolation="nearest", transform=ax.get_xaxis_transform())


# tracking error (and prediction error) along the lap
panels = [("tracking_error", "Tracking error [m]", arms)]
if map_name == "l_shape":
    panels.append(("prediction_error", "Prediction error [m]", arms[1:]))
fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(ps.COLUMN_IN, 1.45 + 1.15 * (len(panels) - 1)), squeeze=False)
for ax, (key, ylabel, panel_arms) in zip(axes[:, 0], panels):
    friction_background(ax)
    for arm in panel_arms:
        med, lo, hi = binned(data[arm], key)
        if arm in ("NMPC", "pooled") and key == "tracking_error":
            ax.fill_between(centers, lo, hi, color=ps.COLORS[arm], alpha=0.15, lw=0)
        ax.plot(centers, med, color=ps.COLORS[arm], ls=ps.LINES[arm], lw=1.0 if map_name == "l_shape" else 0.8, label=ps.LABELS[arm])
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, track_len)
    if key == "prediction_error":
        ax.set_yscale("log")   # the few large peaks would hide the differences between the variants
        ax.set_ylim(0.01, 1.0)
    else:
        ax.set_ylim(0, 1.15 * max(np.nanmax(binned(data[a], key)[0]) for a in panel_arms))
if map_name == "Nuerburgring":   # largest deviation: the hairpin, twice the curvature of anything in the l_shape data
    k = np.nanargmax(binned(data["pooled"], "tracking_error")[0])
    axes[0, 0].annotate("hairpin", xy=(centers[k], binned(data["pooled"], "tracking_error")[0][k]), xytext=(centers[k] + 350, 1.2),
                        fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"), va="center")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=2.0, columnspacing=0.9,
           borderaxespad=0.1)
axes[-1, 0].set_xlabel("Progress on the track [m]")
fig.align_ylabels(axes[:, 0])
fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight()))
ps.save(fig, f"evaluation/02_tracking/tracking_error_{map_name}")

# implied friction on the track map
pooled = data["pooled"]
implied = [[] for _ in range(n_bins)]
for d in pooled:
    ok = (d["lap_n"] >= 1) & np.isfinite(d["implied_mu"]) & (d["implied_dist"] > readable_m)
    for p, v in zip(d["track_progress"][ok], d["implied_mu"][ok]):
        implied[min(int(p // bin_m), n_bins - 1)].append(v)
implied = np.array([np.mean(v) if len(v) >= 3 else np.nan for v in implied])
smooth = int(round(10.0 / bin_m)) if map_name == "l_shape" else 1   # display: running mean over 10 m
if smooth > 1:
    pad = np.r_[implied[-smooth:], implied, implied[:smooth]]
    run = np.array([np.nanmean(pad[i - smooth // 2:i + smooth // 2 + 1]) if np.isfinite(pad[i]) else np.nan for i in range(smooth, smooth + n_bins)])
    implied = run
for z in np.unique(np.round(friction[np.isfinite(friction)], 1)):
    m = np.isclose(friction, z, atol=0.05)
    print(f"true mu {z:.1f}: implied {np.nanmean(implied[m]):.2f} (readable on {np.mean(np.isfinite(implied[m])) * 100:.0f}% of the sections)")

x = np.interp(centers, waypoints[:, 0], waypoints[:, 1])
y = np.interp(centers, waypoints[:, 0], waypoints[:, 2])
dx, dy = np.gradient(x), np.gradient(y)
norm = np.hypot(dx, dy) + 1e-9
nx, ny = -dy / norm, dx / norm
size = max(np.ptp(x), np.ptp(y))
offset = 0.022 * size


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


fig, ax = plt.subplots(figsize=(ps.COLUMN_IN, 2.7 if map_name == "l_shape" else 3.1))
norm_c = plt.Normalize(vmin=ps.FRICTION_MIN, vmax=ps.FRICTION_MAX)
ax.plot(x + offset * nx, y + offset * ny, color="#c8c8c8", lw=2.6, solid_capstyle="butt")   # not readable
for sign, values in ((-1, friction), (1, implied)):
    lc = LineCollection(make_segments(x + sign * offset * nx, y + sign * offset * ny), cmap=ps.FRICTION_CMAP, norm=norm_c)
    lc.set_array(np.ma.masked_invalid(values[:-1]))
    lc.set_linewidth(2.6)
    lc.set_capstyle("butt")
    ax.add_collection(lc)

# start/finish line
t_vec = np.array([dx[0], dy[0]]) / norm[0]
n_vec = np.array([nx[0], ny[0]])
center = np.array([x[0], y[0]])
half_width, n_cols, n_rows = 2.2 * offset, 6, 2
tile_w, tile_h = 2 * half_width / n_cols, 0.8 * offset / n_rows
for row in range(n_rows):
    for col in range(n_cols):
        a, c = -half_width + col * tile_w, -0.4 * offset + row * tile_h
        corners = [center + u * n_vec + v * t_vec for u, v in ((a, c), (a + tile_w, c), (a + tile_w, c + tile_h), (a, c + tile_h))]
        ax.add_patch(Polygon(corners, closed=True, facecolor="black" if (row + col) % 2 == 0 else "white", edgecolor="black",
                             linewidth=0.3, zorder=10))

ax.set_aspect("equal")
ax.autoscale()
ax.set_axis_off()
cbar = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=35)
cbar.set_label(r"Friction $\mu$ (inner ring: track, outer ring: implied by the DPC prediction)")
cbar.outline.set_linewidth(0.4)
fig.tight_layout()
ps.save(fig, f"evaluation/02_tracking/implied_friction_{map_name}")

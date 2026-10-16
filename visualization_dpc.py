"""Paper figures from the evaluation logs (evaluation/02_tracking/run_eval.sh):
evaluation/02_tracking/tracking_error_<map>: tracking error along the lap of DPC (pooled data) and NMPC (median over all evaluated
laps of all seeds, band: min-max) on the friction profile; for l_shape below it the tracking error of DPC with the mismatched
and with the pooled data relative to DPC with the matched data (oracle / inverse runs, as the hardware figure).
FIG_ARMS=NMPC,pooled,oracle,inverse: all four in the upper panel and a prediction-error panel; FIG_DIR: output folder.
evaluation/02_tracking/implied_friction_<map>: track map, inner ring true friction, outer ring the prediction-based implied
friction of the pooled DPC (helpers/implied_friction.py), evidence-weighted over all steps of a 10 m window and all laps:
w = sum <P-B, A-B> / sum |A-B|^2 (A, B: the mu-0.5-only and mu-1.1-only predictions, P: the pooled one), so steps on which
friction does not change the prediction carry no weight. Thin outline only: the two predictions differ by less than 1 cm (rms)
there, friction has no visible effect on the predicted motion.
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
fig_dir = os.environ.get("FIG_DIR", "evaluation/02_tracking")   # where the figures go
arms = ("NMPC", "pooled", "oracle", "inverse") if map_name == "l_shape" else ("NMPC", "pooled")
bin_m = 2.0 if map_name == "l_shape" else 10.0
readable_m = 0.01
window_m = 10.0
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
    for k in ("implied_mu", "implied_dist", "implied_d_low", "implied_d_high"):
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
fig_arms = tuple(os.environ["FIG_ARMS"].split(",")) if os.environ.get("FIG_ARMS") else ("NMPC", "pooled")
panels = [("tracking_error", "Tracking error [m]", fig_arms)]
if map_name == "l_shape" and "oracle" in fig_arms:
    panels.append(("prediction_error", "Prediction error [m]", tuple(a for a in fig_arms if a != "NMPC")))
# as the hardware figure: tracking error of the mismatched and the pooled DPC relative to the matched one (default when both runs exist)
relative = map_name == "l_shape" and all(data.get(a) for a in ("oracle", "inverse")) and os.environ.get("FIG_RELATIVE", "1") == "1"
if relative:
    panels.append(("relative", "Tracking error\nvs matched [cm]", ()))
fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(ps.COLUMN_IN, 1.45 + 1.15 * (len(panels) - 1)), squeeze=False)
for ax, (key, ylabel, panel_arms) in zip(axes[:, 0], panels):
    if key != "relative":   # the relative panel stays white (as the hardware figure), the zones are shown above it
        friction_background(ax)
    if key == "relative":
        k5 = max(int(round(window_m / bin_m)), 1)
        smooth = lambda a: np.array([np.nanmean(a[np.arange(i - k5 // 2, i - k5 // 2 + k5) % n_bins]) for i in range(n_bins)])
        te = {a: binned(data[a], "tracking_error")[0] for a in ("pooled", "oracle", "inverse")}
        dw, dp = smooth(100 * (te["inverse"] - te["oracle"])), smooth(100 * (te["pooled"] - te["oracle"]))
        ax.fill_between(centers, 0, dw, color=ps.COLORS["inverse"], alpha=0.35, lw=0, label="DPC mismatched")
        ax.plot(centers, dw, color=ps.COLORS["inverse"], lw=0.6)
        ax.plot(centers, dp, color=ps.COLORS["pooled"], lw=1.4)
        ax.axhline(0, color="#222222", lw=0.6); ax.set_ylabel(ylabel); ax.set_xlim(0, track_len)
        ax.grid(axis="y", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
        ax.set_ylim(1.15 * min(0, np.nanmin(dw), np.nanmin(dp)), 1.15 * max(np.nanmax(dw), np.nanmax(dp)))
        print(f"vs matched along the lap: mismatched {np.nanmean(dw):+.1f} cm, pooled {np.nanmean(dp):+.1f} cm (lap mean)")
        continue
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
if relative:
    h2, l2 = axes[-1, 0].get_legend_handles_labels(); handles, labels = handles + h2, labels + l2
fig.legend(handles, labels, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.8,
           borderaxespad=0.1)
axes[-1, 0].set_xlabel("Progress on the track [m]")
fig.align_ylabels(axes[:, 0])
fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight()))
ps.save(fig, f"{fig_dir}/tracking_error_{map_name}")

# implied friction on the track map (evidence-weighted)
num, den, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
for d in data["pooled"]:
    ok = (d["lap_n"] >= 1) & np.isfinite(d["implied_dist"])
    b = np.minimum((d["track_progress"][ok] // bin_m).astype(int), n_bins - 1)
    D, dA, dB = d["implied_dist"][ok], d["implied_d_low"][ok], d["implied_d_high"][ok]
    np.add.at(num, b, (dB ** 2 + D ** 2 - dA ** 2) / 2)   # <P-B, A-B> from the three position distances
    np.add.at(den, b, D ** 2); np.add.at(cnt, b, 1)
k = max(int(round(window_m / bin_m)), 1)
win = lambda a: np.array([a[np.arange(i - k // 2, i - k // 2 + k) % n_bins].sum() for i in range(n_bins)])
num_w, den_w, cnt_w = win(num), win(den), win(cnt)
w = np.clip(num_w / np.maximum(den_w, 1e-12), 0, 1)
implied = np.where(np.sqrt(den_w / np.maximum(cnt_w, 1)) > readable_m, 0.5 * w + 1.1 * (1 - w), np.nan)
for z in np.unique(np.round(friction[np.isfinite(friction)], 1)):
    m = np.isclose(friction, z, atol=0.05)
    print(f"true mu {z:.1f}: implied {np.nanmean(implied[m]):.2f} (identifiable on {np.mean(np.isfinite(implied[m])) * 100:.0f}% of the sections)")

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


def track_map(outer, outer_cmap, outer_norm, outer_label, outline_label, stem, outer_ticks=None):
    """track map: inner ring the true friction, outer ring `outer` (thin outline where it is NaN), a colour bar for each"""
    fig = plt.figure(figsize=(ps.COLUMN_IN, 2.5 if map_name == "l_shape" else 3.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.035], hspace=0.08, wspace=0.25, left=0.04, right=0.96, bottom=0.12, top=0.99)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(x + offset * nx, y + offset * ny, color="#555555", lw=0.5)
    rings = []
    for sign, values, cmap, nrm in ((-1, friction, ps.FRICTION_CMAP, plt.Normalize(ps.FRICTION_MIN, ps.FRICTION_MAX)),
                                    (1, outer, outer_cmap, outer_norm)):
        lc = LineCollection(make_segments(x + sign * offset * nx, y + sign * offset * ny), cmap=cmap, norm=nrm)
        lc.set_array(np.ma.masked_invalid(values[:-1])); lc.set_linewidth(2.6); lc.set_capstyle("butt")
        ax.add_collection(lc); rings.append(lc)
    t_vec = np.array([dx[0], dy[0]]) / norm[0]; n_vec = np.array([nx[0], ny[0]]); center = np.array([x[0], y[0]])
    half_width, n_cols, n_rows = 2.2 * offset, 6, 2   # start/finish line
    tile_w, tile_h = 2 * half_width / n_cols, 0.8 * offset / n_rows
    for row in range(n_rows):
        for col in range(n_cols):
            a, c = -half_width + col * tile_w, -0.4 * offset + row * tile_h
            corners = [center + u * n_vec + v * t_vec for u, v in ((a, c), (a + tile_w, c), (a + tile_w, c + tile_h), (a, c + tile_h))]
            ax.add_patch(Polygon(corners, closed=True, facecolor="black" if (row + col) % 2 == 0 else "white", edgecolor="black",
                                 linewidth=0.3, zorder=10))
    ax.set_aspect("equal"); ax.autoscale(); ax.set_axis_off()
    if np.isnan(outer).any():
        ax.plot([], [], color="#555555", lw=0.5, label=outline_label)
        ax.legend(loc="lower left", fontsize=7, handlelength=1.5, borderaxespad=0)
    for col, (lc, label, ticks) in enumerate(((rings[0], r"inner ring: track friction $\mu$", [0.5, 0.8, 1.1]), (rings[1], outer_label, outer_ticks))):
        cbar = fig.colorbar(lc, cax=fig.add_subplot(gs[1, col]), orientation="horizontal", ticks=ticks)
        cbar.set_label(label, labelpad=1, fontsize=7); cbar.outline.set_linewidth(0.4); cbar.ax.tick_params(labelsize=6.5, length=2)
    ps.save(fig, stem)


track_map(implied, ps.FRICTION_CMAP, plt.Normalize(ps.FRICTION_MIN, ps.FRICTION_MAX), r"outer ring: implied by the prediction",
          "friction has no effect on the prediction", f"{fig_dir}/implied_friction_{map_name}")

import json
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
import yaml

map_name = "Nuerburgring" # l_shape, Nuerburgring
name_dpc = "log_dpc_eval_%s" % map_name
name_nmpc = "log_nmpc_eval_%s" % map_name
dataset_frictions = np.array([0.5, 1.1])
window = 3
kernel = np.ones(window) / window

with open("configs/config_%s.yaml" % map_name) as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
waypoints = np.array(raceline)


def load_log(name):
    with open(name, "r") as f:
        data = json.load(f)
    for key in data:
        data[key] = np.array(data[key], dtype=object)
    return data


def get_tracking_stats(data):
    lap_n = data["lap_n"].astype(int)
    valid_laps = np.unique(lap_n)
    valid_laps = valid_laps[valid_laps >= 1]

    track_progress_all = []
    tracking_error_all = []
    friction_all = []

    for lap in valid_laps:
        mask = lap_n == lap

        track_progress = np.asarray(data["track_progress"][mask], dtype=float)
        tracking_error = np.asarray(data["tracking_error"][mask], dtype=float)
        friction = np.asarray(data["friction"][mask], dtype=float)

        order = np.argsort(track_progress)
        track_progress_all.append(track_progress[order])
        tracking_error_all.append(tracking_error[order])
        friction_all.append(friction[order])

    common_progress = np.unique(np.concatenate(track_progress_all))

    tracking_interp = np.array([
        np.interp(common_progress, p, y, left=np.nan, right=np.nan)
        for p, y in zip(track_progress_all, tracking_error_all)
    ])

    friction_interp = np.array([
        np.interp(common_progress, p, y, left=np.nan, right=np.nan)
        for p, y in zip(track_progress_all, friction_all)
    ])

    tracking_min = np.nanmin(tracking_interp, axis=0)
    tracking_med = np.nanmedian(tracking_interp, axis=0)
    tracking_max = np.nanmax(tracking_interp, axis=0)
    friction_mean = np.nanmedian(friction_interp, axis=0)

    return common_progress, tracking_min, tracking_med, tracking_max, friction_mean

DPC_COLOR = "#1b7f3b"   # dark green
NMPC_COLOR = "#222222"  # near-black

data_dpc = load_log(name_dpc)
data_nmpc = load_log(name_nmpc)

lap_n = data_dpc["lap_n"].astype(int)
valid_laps = np.unique(lap_n)
valid_laps = valid_laps[valid_laps >= 1]

track_progress_all = []
tracking_error_all = []
used_mu_all = []
friction_all = []

for lap in valid_laps:
    mask = lap_n == lap

    track_progress = np.asarray(data_dpc["track_progress"][mask], dtype=float)
    tracking_error = np.asarray(data_dpc["tracking_error"][mask], dtype=float)
    friction = np.asarray(data_dpc["friction"][mask], dtype=float)

    usage_raw = data_dpc["usage_fraction"][mask]
    used_mu = np.array([
        np.nan if u is None else np.asarray(u, dtype=float) @ dataset_frictions
        for u in usage_raw
    ])

    order = np.argsort(track_progress)
    track_progress_all.append(track_progress[order])
    tracking_error_all.append(tracking_error[order])
    used_mu_all.append(used_mu[order])
    friction_all.append(friction[order])

common_progress = np.unique(np.concatenate(track_progress_all))

tracking_interp = np.array([
    np.interp(common_progress, p, y, left=np.nan, right=np.nan)
    for p, y in zip(track_progress_all, tracking_error_all)
])

used_mu_interp = np.array([
    np.interp(common_progress, p, y, left=np.nan, right=np.nan)
    for p, y in zip(track_progress_all, used_mu_all)
])

friction_interp = np.array([
    np.interp(common_progress, p, y, left=np.nan, right=np.nan)
    for p, y in zip(track_progress_all, friction_all)
])

friction_mean = np.nanmedian(friction_interp, axis=0)
used_mu_mean = np.nanmean(used_mu_interp, axis=0)

tracking_min = np.nanmin(tracking_interp, axis=0)
tracking_med = np.nanmedian(tracking_interp, axis=0)
tracking_max = np.nanmax(tracking_interp, axis=0)

progress_nmpc, tracking_min_nmpc, tracking_med_nmpc, tracking_max_nmpc, _ = get_tracking_stats(data_nmpc)

wp_s = waypoints[:, 0]
wp_x = waypoints[:, 1]
wp_y = waypoints[:, 2]

x_map = np.interp(common_progress, wp_s, wp_x)
y_map = np.interp(common_progress, wp_s, wp_y)

used_mu_plot = np.convolve(used_mu_mean, kernel, mode="same")
vmin = 0.5
vmax = 1.1

fig, ax = plt.subplots(figsize=(10, 3))
friction_img = friction_mean[np.newaxis, :]
ax.imshow(
    friction_img,
    extent=[common_progress[0], common_progress[-1], 0, 1],
    aspect="auto",
    cmap="coolwarm",
    vmin=0.5,
    vmax=1.1,
    alpha=0.5,
    origin="lower",
    interpolation="nearest",
    transform=ax.get_xaxis_transform(),
)

ax.fill_between(common_progress, tracking_min, tracking_max, alpha=0.250, color=DPC_COLOR)
ax.plot(common_progress, tracking_med, color=DPC_COLOR, label="DPC")

ax.fill_between(progress_nmpc, tracking_min_nmpc, tracking_max_nmpc, alpha=0.250, color=NMPC_COLOR)
ax.plot(progress_nmpc, tracking_med_nmpc, color=NMPC_COLOR, label="NMPC")

ax.set_xlabel("Progress on the track [m]")
ax.set_ylabel("Tracking error [m]")
ax.legend()
plt.show()

fig.savefig(f"evaluation/02_tracking/tracking_error_{map_name}.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"evaluation/02_tracking/tracking_error_{map_name}.svg", bbox_inches="tight")

fig, ax = plt.subplots()
friction_img = friction_mean[np.newaxis, :]
ax.imshow(
    friction_img,
    extent=[common_progress[0], common_progress[-1], 0, 1],
    aspect="auto",
    cmap="coolwarm",
    vmin=0.5,
    vmax=1.1,
    alpha=0.25,
    origin="lower",
    interpolation="nearest",
    transform=ax.get_xaxis_transform(),
)

ax.plot(common_progress, used_mu_mean, label="implied friction from usage")
ax.plot(common_progress, friction_mean, label="true friction")
ax.set_xlabel("track progress")
ax.set_ylabel("friction")
ax.legend()
plt.show()

from matplotlib.collections import LineCollection

dx = np.gradient(x_map)
dy = np.gradient(y_map)
norm = np.sqrt(dx**2 + dy**2) + 1e-9

nx = -dy / norm
ny = dx / norm

offset = 0.0

x_used = x_map + offset * nx
y_used = y_map + offset * ny

x_friction = x_map - offset * nx
y_friction = y_map - offset * ny

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

segments_used = make_segments(x_used, y_used)
segments_friction = make_segments(x_friction, y_friction)

norm_c = plt.Normalize(vmin=0.5, vmax=1.1)

fig, ax = plt.subplots()

ax.plot(x_map, y_map, color="gray", linewidth=1, alpha=0.5)

lc_used = LineCollection(
    segments_used,
    cmap="coolwarm",
    norm=norm_c,
)
lc_used.set_array(used_mu_plot[:-1])
lc_used.set_linewidth(5)
lc_used.set_capstyle("round")
lc_used.set_joinstyle("round")
ax.add_collection(lc_used)

lc_friction = LineCollection(
    segments_friction,
    cmap="coolwarm",
    norm=norm_c,
)
lc_friction.set_array(friction_mean[:-1])
lc_friction.set_linewidth(5)
lc_friction.set_capstyle("round")
lc_friction.set_joinstyle("round")
ax.add_collection(lc_friction)

plt.colorbar(lc_friction, ax=ax, label="friction / implied friction")

from matplotlib.patches import Polygon

i0 = 0

tx = dx[i0] / norm[i0]
ty = dy[i0] / norm[i0]

nx0 = -ty
ny0 = tx

line_half_width = 6.0
n_cols = 8
n_rows = 2

tile_w = 2 * line_half_width / n_cols
tile_h = 4.0 / n_rows

center = np.array([x_map[i0], y_map[i0]])
t_vec = np.array([tx, ty])
n_vec = np.array([nx0, ny0])

for row in range(n_rows):
    for col in range(n_cols):
        a = -line_half_width + col * tile_w
        b = a + tile_w

        c = -2.0 + row * tile_h
        d = c + tile_h

        p1 = center + a * n_vec + c * t_vec
        p2 = center + b * n_vec + c * t_vec
        p3 = center + b * n_vec + d * t_vec
        p4 = center + a * n_vec + d * t_vec

        color = "black" if (row + col) % 2 == 0 else "white"
        ax.add_patch(
            Polygon(
                [p1, p2, p3, p4],
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=10,
            )
        )

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
ax.autoscale()
plt.show()

fig.savefig(f"evaluation/02_tracking/implied_friction_{map_name}.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"evaluation/02_tracking/implied_friction_{map_name}.svg", bbox_inches="tight")
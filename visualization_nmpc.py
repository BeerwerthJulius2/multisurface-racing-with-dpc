import json
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
import yaml

map_name = "l_shape"
name = "log_nmpc_eval_%s" % map_name
dataset_idx = 0   # choose which usage_fraction entry to plot
window = 15
kernel = np.ones(window) / window

with open('configs/config_%s.yaml' % map_name) as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
waypoints = np.array(raceline)

with open(name, "r") as f:
    data = json.load(f)

for key in data:
    data[key] = np.array(data[key], dtype=object)

lap_n = data["lap_n"].astype(int)
valid_laps = np.unique(lap_n)
valid_laps = valid_laps[valid_laps >= 1]

track_progress_all = []
tracking_error_all = []
usage_all = []
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

usage_interp = np.array([
    np.interp(common_progress, p, y, left=np.nan, right=np.nan)
    for p, y in zip(track_progress_all, usage_all)
])

friction_interp = np.array([
    np.interp(common_progress, p, y, left=np.nan, right=np.nan)
    for p, y in zip(track_progress_all, friction_all)
])

friction_mean = np.nanmedian(friction_interp, axis=0)

tracking_min = np.nanmin(tracking_interp, axis=0)
tracking_med = np.nanmedian(tracking_interp, axis=0)
tracking_max = np.nanmax(tracking_interp, axis=0)

wp_s = waypoints[:, 0]
wp_x = waypoints[:, 1]
wp_y = waypoints[:, 2]

x_map = np.interp(common_progress, wp_s, wp_x)
y_map = np.interp(common_progress, wp_s, wp_y)

fr_min = np.min(friction_mean)
fr_max = np.max(friction_mean)

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

ax.fill_between(common_progress, tracking_min, tracking_max, alpha=0.25)
ax.plot(common_progress, tracking_med)
ax.set_xlabel("track progress")
ax.set_ylabel("tracking error")
plt.show()
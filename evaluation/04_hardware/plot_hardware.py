"""Hardware figure of the paper (evaluation/04_hardware/hardware.pdf/.svg) from hardware_runs.json (extract_lab_runs.py), in the
style of Fig. 1: per DPC variant (rows) and tyre set (colour: slippery = low friction, normal = high friction) the distribution
over all steps of the 3 runs of the lateral error and the prediction error (rms over the 20-step horizon), and over all
complete laps of the lap time. Also prints the medians.
usage (repo root): python evaluation/04_hardware/plot_hardware.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.getcwd())
from helpers import paper_style as ps   # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
runs = json.load(open(os.path.join(here, "hardware_runs.json")))
ps.apply()
arms = (("oracle", "Oracle"), ("inverse", "Inverse"), ("pooled", "Pooled"))
tyres = (("slippery", "Slippery tyres", ps.friction_color(0.5)), ("normal", "Normal tyres", ps.friction_color(1.1)))
panels = (("lat_steps_cm", "Lateral error [cm]"), ("pred_steps_cm", "Prediction error [cm]"), ("lap_times_s", "Lap time [s]"))
fig, axes = plt.subplots(1, 3, figsize=(ps.COLUMN_IN, 1.75), sharey=True)
for ax, (key, label) in zip(axes, panels):
    groups = []
    for arm, arm_label in arms:
        boxes = []
        for tyre, _, color in tyres:
            v = np.concatenate([r[key] for r in runs if r["tyre"] == tyre and r["arm"] == arm])
            boxes.append((v, color)); print(f"{key:14s} {tyre:8s} {arm:8s} median {np.median(v):6.2f}  n {len(v)}")
        groups.append((arm_label, boxes))
    ps.box_rows(ax, groups)
    ax.set_xlabel(label)
    if key != "lap_times_s": ax.set_xlim(left=0)
for ax in axes[1:]: ax.tick_params(axis="y", labelleft=False)
fig.legend(handles=[mpatches.Patch(facecolor=c, edgecolor="black", lw=0.6, label=l) for _, l, c in tyres], ncol=2,
           loc="upper center", bbox_to_anchor=(0.55, 1.0), handlelength=1.2, columnspacing=1.2, borderaxespad=0.1)
fig.tight_layout(w_pad=0.5, rect=(0, 0, 1, 1 - 0.16 / fig.get_figheight()))
ps.save(fig, os.path.join(here, "hardware"))

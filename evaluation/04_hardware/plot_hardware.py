"""Hardware figure of the paper (evaluation/04_hardware/hardware.pdf/.svg) from hardware_runs.json (extract_lab_runs.py):
per tyre set and DPC variant the lateral error, the prediction error (rms over the 20-step horizon) and the lap time of
each run (dots) and their mean (bar). Also prints the means.
usage (repo root): python evaluation/04_hardware/plot_hardware.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from helpers import paper_style as ps   # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
runs = json.load(open(os.path.join(here, "hardware_runs.json")))
ps.apply()
arms, tyres = ("pooled", "oracle", "inverse"), ("slippery", "normal")
panels = (("lat_cm", "Lateral error [cm]"), ("pred_rms_cm", "Prediction error [cm]"), ("lap_s", "Lap time [s]"))
fig, axes = plt.subplots(1, 3, figsize=(ps.COLUMN_IN, 1.55))
for ax, (key, label) in zip(axes, panels):
    for gi, tyre in enumerate(tyres):
        for ai, arm in enumerate(arms):
            v = np.array([r[key] for r in runs if r["tyre"] == tyre and r["arm"] == arm]); x = gi * 3.6 + ai
            ax.plot([x - 0.38, x + 0.38], [v.mean()] * 2, color=ps.COLORS[arm], lw=1.6, solid_capstyle="butt",
                    label=ps.LABELS[arm] if gi == 0 else None)
            ax.scatter(x + np.array([-0.18, 0, 0.18])[:len(v)], v, s=6, color=ps.COLORS[arm], zorder=3, linewidths=0)
            print(f"{key:12s} {tyre:8s} {arm:8s} mean {v.mean():6.2f}  runs " + " ".join(f"{u:.2f}" for u in v))
    ax.set_xticks([1, 4.6]); ax.set_xticklabels(["slippery\ntyres", "normal\ntyres"]); ax.tick_params(axis="x", length=0)
    ax.set_title(label, pad=3); ax.set_xlim(-0.7, 6.3)
    ax.grid(axis="y", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
    if key != "lap_s": ax.set_ylim(bottom=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 1 - 0.16 / fig.get_figheight()))
ps.save(fig, os.path.join(here, "hardware"))

"""Hardware figure of the paper (evaluation/04_hardware/hardware.pdf/.svg) from hardware_runs.json (extract_lab_runs.py), in the
style of the closed-loop figures (helpers/paper_style.dot_bars): per tyre set the DPC variants side by side, one dot per run and a
bar at the mean, for the lateral error (rms), the prediction error (rms over the 20-step horizon, median over the run) and the
lap time. Also prints the values.
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
arms, tyres = ("oracle", "inverse", "pooled"), ("slippery", "normal")
panels = (("lat_cm", "Lateral error [cm]"), ("pred_rms_cm", "Prediction error [cm]"), ("lap_s", "Lap time [s]"))
fig, axes = plt.subplots(1, 3, figsize=(ps.COLUMN_IN, 1.6))
for ax, (key, label) in zip(axes, panels):
    values = {(t, a): [r[key] for r in runs if r["tyre"] == t and r["arm"] == a] for t in tyres for a in arms}
    for (t, a), v in values.items():
        print(f"{key:12s} {t:8s} {a:8s} mean {np.mean(v):6.2f}  runs " + " ".join(f"{u:.2f}" for u in v))
    ps.dot_bars(ax, tyres, arms, values)
    ax.set_xticklabels(["slippery\ntyres", "normal\ntyres"]); ax.set_title(label, pad=3)
    if key != "lap_s": ax.set_ylim(bottom=0)
handles = [plt.Line2D([], [], color=ps.COLORS[a], lw=1.6, marker="o", ms=2.5, label=ps.LABELS[a]) for a in arms]
fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.0, borderaxespad=0.1)
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 1 - 0.16 / fig.get_figheight()))
ps.save(fig, os.path.join(here, "hardware"))

"""Hardware figure of the paper (evaluation/04_hardware/hardware.pdf/.svg) from hardware_runs.json (extract_lab_runs.py), as the
closed-loop figures (helpers/paper_style.dumbbell): per tyre set the bar from the matched to the mismatched DPC, pooled as the large
dot; means over the 3 runs of the lateral error (rms), the prediction error (rms over the 20-step horizon, median over the run)
and the lap time. Also prints the values.
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
arms, tyres = ("pooled", "oracle", "inverse"), (("slippery", "slippery\ntyres"), ("normal", "normal\ntyres"))
panels = (("lat_cm", "Lateral error [cm]"), ("pred_rms_cm", "Prediction error [cm]"), ("lap_s", "Lap time [s]"))
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(ps.COLUMN_IN, 1.35))
for ax, (key, label) in zip(axes, panels):
    values = {}
    for t, name in tyres:
        for a in arms:
            v = [r[key] for r in runs if r["tyre"] == t and r["arm"] == a]
            values[(name, a)] = float(np.mean(v)); print(f"{key:12s} {t:8s} {a:8s} {np.mean(v):6.2f}  runs " + " ".join(f"{u:.2f}" for u in v))
    lo, hi = min(values.values()), max(values.values())
    ps.dumbbell(ax, [n for _, n in tyres], values, xmax=None if key == "lap_s" else 1.15 * hi)
    if key == "lap_s":
        ax.set_xlim(lo - 0.15, hi + 0.15)
    ax.set_xlabel(label)
ps.arm_legend(fig, arms)
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight()))
ps.save(fig, os.path.join(here, "hardware"))

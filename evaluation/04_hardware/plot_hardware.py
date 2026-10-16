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

# along the lap (as the simulation's tracking figure): per tyre set the lateral error of the mismatched and the pooled run
# relative to the matched run, per track position (median over the steps of the 3 runs, 0.25 m sections, 1 m running mean)
L = runs[0]["lap_length_m"]; bw = 0.25; nbin = int(np.ceil(L / bw)); c = (np.arange(nbin) + 0.5) * bw
def along(t, a):
    v = [[] for _ in range(nbin)]
    for r in runs:
        if r["tyre"] == t and r["arm"] == a:
            for s_, e in zip(r["lat_steps_s_m"], r["lat_steps_cm"]):
                v[min(int(s_ // bw), nbin - 1)].append(e)
    med = np.array([np.median(x) if x else np.nan for x in v]); k = 4
    return np.array([np.nanmean(med[np.arange(i - k // 2, i - k // 2 + k) % nbin]) for i in range(nbin)])
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(ps.COLUMN_IN, 2.35))
for ax, (t, name) in zip(axes, (("slippery", "slippery tyres"), ("normal", "normal tyres"))):
    m_, w_, p_ = along(t, "oracle"), along(t, "inverse"), along(t, "pooled")
    dw, dp = w_ - m_, p_ - m_
    ax.fill_between(c, 0, dw, color=ps.COLORS["inverse"], alpha=0.3, lw=0, label="mismatched")
    ax.plot(c, dw, color=ps.COLORS["inverse"], lw=0.6); ax.plot(c, dp, color=ps.COLORS["pooled"], lw=1.4, label="pooled")
    ax.axhline(0, color="#222222", lw=0.6); ax.set_xlim(0, L); ax.set_ylabel("Lateral error\nvs matched [cm]")
    ax.set_title(name, loc="left", fontsize=7, pad=2)
    ax.grid(axis="y", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
    print(f"along the lap, {name}: mismatched {np.nanmean(dw):+.2f} cm, pooled {np.nanmean(dp):+.2f} cm vs matched")
axes[1].legend(loc="upper right", fontsize=7, title="vs matched data", title_fontsize=6.5, ncol=2)
axes[-1].set_xlabel("Progress on the track [m]")
fig.tight_layout(h_pad=0.3)
ps.save(fig, os.path.join(here, "hardware_along"))

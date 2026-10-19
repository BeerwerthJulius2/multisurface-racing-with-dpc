"""Hardware figure of the paper (evaluation/04_hardware/hardware.pdf/.svg) from hardware_runs.json (extract_lab_runs.py): per tyre
set the lateral error of DPC with the mismatched and with the pooled data relative to DPC with the matched data along the lap,
as the lower panel of the simulation's tracking figure. Also prints the lap means.
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
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(ps.COLUMN_IN, 1.75))
for ax, (t, name) in zip(axes, (("slippery", "low-friction tires"), ("normal", "standard tires"))):
    m_, w_, p_ = along(t, "oracle"), along(t, "inverse"), along(t, "pooled")
    dw, dp = w_ - m_, p_ - m_
    ax.fill_between(c, 0, dw, color=ps.COLORS["inverse"], alpha=0.3, lw=0, label="DPC mismatched")
    ax.plot(c, dw, color=ps.COLORS["inverse"], lw=0.6); ax.plot(c, dp, color=ps.COLORS["pooled"], lw=1.4, label="DPC pooled")
    h, l = ax.get_legend_handles_labels()   # pooled first, as in the tables
    ax.legend(h[::-1], l[::-1], loc="upper right", ncol=2, handlelength=1.8, columnspacing=0.8, borderaxespad=0.2)
    ax.axhline(0, color="#222222", lw=0.6); ax.set_xlim(0, L)
    ax.set_title(name, loc="left", fontsize=7, pad=2)
    ax.grid(axis="y", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
    print(f"along the lap, {name}: mismatched {np.nanmean(dw):+.2f} cm, pooled {np.nanmean(dp):+.2f} cm vs matched")
axes[-1].set_xlabel("Progress on the track [m]")
fig.tight_layout(h_pad=0.3, rect=(0.035, 0, 1, 1))
fig.text(0.005, 0.55, "Lateral error vs matched [cm]", rotation=90, ha="left", va="center")   # one label for both panels
ps.save(fig, os.path.join(here, "hardware"))

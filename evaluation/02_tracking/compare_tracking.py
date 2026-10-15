"""Median tracking error along the lap (laps >= 1, all runs of a set) per friction zone, lap time and speed vs reference, for
sets of sim logs. usage (sim repo dir): compare_runs.py "<name>=<glob>" ..."""
import json, glob, sys, numpy as np
g = np.arange(0, 696, 2.0); nb = len(g)
print(f"{'set':38s} {'runs':>4s} {'laps':>4s} {'lap s':>6s} | median error along the lap: lap / mu 0.5 / 0.8 / 1.1 [m] | speed - ref mean 0.5 / 0.8 / 1.1")
for a in sys.argv[1:]:
    name, pat = a.split("=", 1); files = sorted(glob.glob(pat))
    if not files: print(f"{name:38s} no logs"); continue
    acc = [[] for _ in range(nb)]; fr = [[] for _ in range(nb)]; LT = []; DV = {z: [] for z in (0.5, 0.8, 1.1)}; nl = 0
    for f in files:
        d = json.load(open(f)); lap = np.array(d["lap_n"]); p = np.array(d["track_progress"], float); te = np.array(d["tracking_error"], float); t = np.array(d["time"], float)
        frc = np.round(np.array(d["friction"], float), 1); dv = np.array(d["vx"], float) - np.array(d["v_ref"], float)
        full = [l for l in sorted(set(lap[lap >= 1])) if (lap == l).sum() > 0.9 * (lap == 1).sum()]; nl += len(full)
        LT += [t[lap == l].max() - t[lap == l].min() for l in full]
        for k in np.where(lap >= 1)[0]:
            b = min(int(p[k] // 2), nb - 1); acc[b].append(te[k]); fr[b].append(frc[k])
            if frc[k] in DV: DV[frc[k]].append(dv[k])
    M = np.array([np.median(v) if v else np.nan for v in acc]); F = np.array([np.median(v) if v else np.nan for v in fr])
    print(f"{name:38s} {len(files):4d} {nl:4d} {np.mean(LT):6.2f} | {np.nanmean(M):.3f} / " + " / ".join(f"{np.nanmean(M[np.isclose(F, z)]):.3f}" for z in (0.5, 0.8, 1.1)) + " | " + " / ".join(f"{np.mean(DV[z]):+.2f}" for z in (0.5, 0.8, 1.1)))

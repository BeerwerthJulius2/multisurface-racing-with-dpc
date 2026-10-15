"""Sim evaluation of pooled / oracle / inverse runs (DPC.py logs, laps >= 1): per run and friction zone the tracking error,
the time spent in the zone per lap, speed - reference, the prediction error (logged planned positions vs the positions
reached; horizon offset fitted on the first predicted point), and the Eq. 11 share of the mu-0.5 data vs its shuffle null.
usage (sim repo dir): sim_eval.py <log> ...   (arm and seed from the file name ..._eval_<arm>_s<seed>)"""
import json, re, sys, numpy as np
def L(f):
    d = json.load(open(f)); return {k: d[k] for k in d}
rows = []
for f in sys.argv[1:]:
    d = L(f); arm, seed = re.search(r"_eval_([a-z]+)_s(\d+)$", f).groups()
    lap = np.array(d["lap_n"], int); fr = np.round(np.array(d["friction"], float), 1); te = np.array(d["tracking_error"], float)
    vx = np.array(d["vx"], float); vr = np.array(d["v_ref"], float); x = np.array(d["x"], float); y = np.array(d["y"], float); t = np.array(d["time"], float)
    nul = lambda k: np.array([np.nan if v is None else v for v in d[k]], float)
    el, en = nul("eta_low"), nul("eta_low_null")
    P = d["pred_xy"]; H = next(len(p[0]) for p in P if p is not None)
    best = None
    for off in (-1, 0, 1):   # which logged step the first predicted point belongs to
        e0 = [np.hypot(p[0][0] - x[k + off], p[1][0] - y[k + off]) for k, p in enumerate(P) if p is not None and 0 <= k + off < len(x)]
        if best is None or np.median(e0) < best[1]: best = (off, np.median(e0))
    off = best[0]; eH = np.full(len(x), np.nan)
    for k, p in enumerate(P):
        if p is not None and k + off + H - 1 < len(x):
            eH[k] = np.sqrt(np.mean([(p[0][j] - x[k + off + j]) ** 2 + (p[1][j] - y[k + off + j]) ** 2 for j in range(H)]))
    valid = lap >= 1; laps = sorted(set(lap[valid]) - {lap.max()}) if (lap == lap.max()).sum() < (lap == 1).sum() * 0.9 else sorted(set(lap[valid]))
    lt = [t[lap == l].max() - t[lap == l].min() + 0.1 for l in laps]
    r = dict(arm=arm, seed=int(seed), laps=len(laps), lap_time=np.mean(lt), off=off, done=len(laps) >= 10)
    for z in (0.5, 0.8, 1.1):
        m = valid & (fr == z)
        r[z] = dict(te=np.mean(te[m]), te95=np.percentile(te[m], 95), dt=m.sum() * 0.1 / max(len(laps), 1), dv=np.mean(vx[m] - vr[m]), pred=np.nanmedian(eH[m]),
                    eta=np.nanmean(el[m]), null=np.nanmean(en[m]))
    r["all"] = dict(te=np.mean(te[valid]), pred=np.nanmedian(eH[valid]))
    rows.append(r)
print(f"prediction horizon {H} steps, first predicted point = logged step k{rows[0]['off']:+d}")
print(f"{'arm':8s} {'seed':>4s} {'laps':>4s} {'lap s':>6s} | per zone mu 0.5 / 0.8 / 1.1: tracking error mean [m] | zone time per lap [s] | speed - ref [m/s] | pred. error [m] | eta_0.5 - null")
for r in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
    z = (0.5, 0.8, 1.1); g = lambda k, fmt: " / ".join(fmt % r[q][k] for q in z)
    print(f"{r['arm']:8s} {r['seed']:4d} {r['laps']:4d} {r['lap_time']:6.2f} | {g('te','%.3f')} | {g('dt','%.2f')} | {g('dv','%+.2f')} | {g('pred','%.3f')} | " + " / ".join("%+.2f" % (r[q]['eta'] - r[q]['null']) for q in z))
print("\nmean over seeds [min-max]:")
for arm in ("pooled", "oracle", "inverse"):
    R = [r for r in rows if r["arm"] == arm]
    if not R: continue
    s = f"  {arm:8s} n {len(R)} lap {np.mean([r['lap_time'] for r in R]):.2f} s"
    for z in (0.5, 0.8, 1.1):
        v = [r[z]["te"] for r in R]; s += f" | mu {z}: err {np.mean(v):.3f} [{min(v):.3f}-{max(v):.3f}] time {np.mean([r[z]['dt'] for r in R]):.2f} pred {np.mean([r[z]['pred'] for r in R]):.3f}"
    print(s)

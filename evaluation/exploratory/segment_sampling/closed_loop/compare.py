"""Diagnostic (2026-10): compare sampling variants on l_shape from the logs in ../logs.

Usage: .venv/bin/python evaluation/exploratory/segment_sampling/closed_loop/compare.py "" _slide40x6 _tini2 ...
"" is the baseline (contextual sampling); its multi-friction run is the _labcheck log.
"""
import json, os, sys, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
LOGS = os.path.join(ROOT, 'evaluation/exploratory/segment_sampling/logs')
os.chdir(ROOT)
sys.path.insert(0, 'evaluation/exploratory/uniform_friction')
import analyze_uniform_friction as a
F = np.array([0.5, 1.1]); s_ref, k_ref = a.load_curvature('l_shape')
TAGS = sys.argv[1:] or ['', '_slide25x4', '_slide40x6']

def implied(d, key):
    out = []
    for u, l in zip(d.get(key, [None] * len(d['time'])), d['unique_labels']):
        if u is None or l is None: out.append(np.nan); continue
        w = np.zeros(2); w[np.array(l, int)] = u; out.append(w @ F)
    return np.array(out)

def load(n):
    if not os.path.exists(n): return None
    d = json.load(open(n)); k = np.array(d['lap_n']) >= 1
    p, v = np.array(d['track_progress'], float), np.array(d['vx'], float)
    lat = np.abs(v ** 2 * np.interp(p, s_ref, k_ref, period=s_ref[-1]))
    return dict(pool=implied(d, 'data_fraction')[k], usage=implied(d, 'usage_fraction')[k],
                te=np.array(d['tracking_error'], float)[k], fr=np.array(d['friction'], float)[k], lat=lat[k],
                laps=max(d['lap_n']))

print(f"{'variant':<16} | {'uniform: tracking':>18} {'pool sep all/corner':>20} {'usage sep all/corner':>21} | "
      f"{'multi: laps':>11} {'tracking':>8} {'pool corr all/corner':>21} {'usage corr all/corner':>22}")
for tag in TAGS:
    u = {mu: load(f'{LOGS}/log_dpc_eval_l_shape_uniform_{mu}{tag}') for mu in ['0.5', '1.1']}
    m = load(f'{LOGS}/log_dpc_eval_l_shape{"_labcheck" if tag == "" else tag}')
    cells = []
    if all(u.values()):
        c5, c11 = u['0.5']['lat'] > 3.5, u['1.1']['lat'] > 3.5
        sep = lambda key, a5, a11: (np.nanmean(u['1.1'][key][a11]) - np.nanmean(u['0.5'][key][a5])) / 0.6
        allm = lambda r: np.ones_like(r['lat'], bool)
        cells.append(f"{u['0.5']['te'].mean():.3f}/{u['1.1']['te'].mean():.3f}".rjust(18))
        for key in ['pool', 'usage']:
            cells.append(f"{sep(key, allm(u['0.5']), allm(u['1.1'])):+.0%} / {sep(key, c5, c11):+.0%}".rjust(20 if key == 'pool' else 21))
    else:
        cells.append("missing".rjust(61))
    if m:
        cells.append("|" + f"{m['laps']:.0f}".rjust(12) + f"{m['te'].mean():.3f}".rjust(9))
        for key in ['pool', 'usage']:
            x = m[key]; ok = ~np.isnan(x); c = ok & (m['lat'] > 3.5)
            cr = lambda mask: np.corrcoef(x[mask], m['fr'][mask])[0, 1] if np.std(x[mask]) > 0 else 0.0
            cells.append(f"{cr(ok):.2f} / {cr(c):.2f}".rjust(21 if key == 'pool' else 22))
    print(f"{tag or 'off':<16} | " + " ".join(cells))

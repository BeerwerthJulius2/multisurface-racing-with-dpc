"""Table I of the paper (LaTeX, evaluation/02_tracking/table_tracking.tex): tracking and prediction error per friction zone on
l_shape and the lap mean on the Nuerburgring, tracking error with the measure of the figures of visualization_dpc.py
(median over all evaluated laps of all seeds per progress bin, averaged over the bins of a zone / the lap), prediction error
as the median over all steps of a zone (PRED=bins: averaged like the tracking error), in cm. Also prints the seed-to-seed range.
usage (repo root): python evaluation/02_tracking/table_tracking.py [results dir (evaluation/02_tracking/results)]"""
import glob, json, os, sys, warnings
import numpy as np
import yaml

results = sys.argv[1] if len(sys.argv) > 1 else "evaluation/02_tracking/results"
PRED_BINS = os.environ.get("PRED", "steps") == "bins"
warnings.filterwarnings("ignore", "Mean of empty slice")
sys.path.insert(0, os.getcwd())
from helpers.eval_metrics import prediction_error   # noqa: E402  (same measure as the figure)


def curves(map_name, files, bin_m):
    wp = np.loadtxt(yaml.safe_load(open(f"configs/config_{map_name}.yaml"))["wpt_path"], delimiter=";", skiprows=3)
    n = int(np.ceil(wp[-1, 0] / bin_m)); te, pe, fr = ([[] for _ in range(n)] for _ in range(3)); steps = {}
    for f in files:
        d = json.load(open(f)); nan = lambda k: np.array([np.nan if v is None else v for v in d[k]], float)
        lap, p, x, y = nan("lap_n"), nan("track_progress"), nan("x"), nan("y")
        e = prediction_error(d["pred_xy"], x, y) if "pred_xy" in d else np.full(len(x), np.nan)
        for k in np.where(lap >= 1)[0]:
            b = min(int(p[k] // bin_m), n - 1); te[b].append(d["tracking_error"][k]); fr[b].append(d["friction"][k])
            if np.isfinite(e[k]):
                pe[b].append(e[k]); steps.setdefault(round(d["friction"][k], 1), []).append(e[k])
    med = lambda L: np.array([np.median(v) if v else np.nan for v in L])
    return med(te), med(pe), np.round(med(fr), 1), steps


def zones(map_name, files, bin_m=2.0):
    te, pe, fr, steps = curves(map_name, files, bin_m)
    if PRED_BINS:
        z = {mu: (np.nanmean(te[fr == mu]), np.nanmean(pe[fr == mu])) for mu in (0.5, 0.8, 1.1)}
        z["lap"] = (np.nanmean(te), np.nanmean(pe)); return z
    med = lambda v: np.median(v) if len(v) else np.nan
    z = {mu: (np.nanmean(te[fr == mu]), med(steps.get(mu, []))) for mu in (0.5, 0.8, 1.1)}
    z["lap"] = (np.nanmean(te), med(sum(steps.values(), []))); return z


arms = ("pooled", "oracle", "inverse", "NMPC")
names = {"NMPC": "NMPC", "pooled": "DPC pooled", "oracle": "DPC matched", "inverse": "DPC mismatched"}
pat = lambda m, a: os.path.join(results, f"log_nmpc_eval_{m}_s*" if a == "NMPC" else f"log_dpc_eval_{m}_eval_{a}_s*")
L = {a: zones("l_shape", sorted(glob.glob(pat("l_shape", a)))) for a in arms}
N = {a: curves("Nuerburgring", sorted(glob.glob(pat("Nuerburgring", a))), 10.0)[0] for a in ("NMPC", "pooled")}
for a in arms:   # seed-to-seed range of the lap tracking error
    per_seed = [zones("l_shape", [f])["lap"][0] for f in sorted(glob.glob(pat("l_shape", a)))]
    print(f"{a:8s} l_shape lap tracking error per seed [cm]: " + " ".join(f"{100 * v:.1f}" for v in per_seed))

cols = [(0.5, 0), (0.8, 0), (1.1, 0), ("lap", 0), (0.5, 1), (0.8, 1), (1.1, 1)]
best = {c: min(L[a][c[0]][c[1]] for a in arms if np.isfinite(L[a][c[0]][c[1]])) for c in cols}
fmt = lambda v, b: "--" if not np.isfinite(v) else (r"\textbf{%.1f}" if np.isclose(v, b) else "%.1f") % (100 * v)
rows = []
for a in arms:
    cells = [fmt(L[a][z][i], best[(z, i)]) for z, i in cols]
    nbr = "%.1f" % (100 * np.nanmean(N[a])) if a in N else "--"
    rows.append(f"{names[a]} & " + " & ".join(cells[:4]) + " & " + " & ".join(cells[4:]) + f" & {nbr} \\\\")
tex = r"""\begin{table}[!t]
\caption{Tracking and prediction error in cm (30 laps, 3 seeds; best per column in bold).}
\label{tab:tracking}
\centering
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{l cccc ccc c}
\hline
 & \multicolumn{4}{c}{Tracking error, L-shape} & \multicolumn{3}{c}{Prediction error} & Tracking \\
 & \multicolumn{3}{c}{zone $\mu$} & & \multicolumn{3}{c}{zone $\mu$} & Nürburgring \\
\cline{2-4}\cline{6-8}
 & 0.5 & 0.8 & 1.1 & lap & 0.5 & 0.8 & 1.1 & lap \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
out = "evaluation/02_tracking/table_tracking.tex"
open(out, "w").write(tex); print(tex); print("written", out)

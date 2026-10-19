"""Fig. 1 of the paper (evaluation/01_prediction/prediction_error.pdf/.svg): open-loop prediction error on held-out windows of the
random-excitation data (data/dpc_experiments/01_prediction_v2, mu 0.5 and 1.1), with 300 contextually sampled trajectories of the
pooled, the matched and the mismatched data (prediction_results_random.json from prediction_error_eval.py with
PRED_DATA=data/dpc_experiments/01_prediction_v2 PRED_LABELS=0:mu_05,1:mu_11 PRED_CACHE=0 PRED_TINI=10
PRED_KEYS=matched_contextual,mismatched_contextual,pooled_contextual PRED_OUT=evaluation/01_prediction/prediction_results_random.json).
One panel per evaluation friction, the datasets as rows (pooled, matched, mismatched from the top); dashed: matched median.
Boxes: quartiles and median, whiskers 10th to 90th percentile.
usage (repo root): python evaluation/01_prediction/visualize_prediction_errors.py evaluation/01_prediction/prediction_results_random.json"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

sys.path.insert(0, os.getcwd())
from helpers import paper_style as ps  # noqa: E402  (run from the repo root)

ROWS = [("pooled_contextual", "pooled"), ("matched_contextual", "matched"), ("mismatched_contextual", "mismatched")]   # top to bottom
XMAX = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="prediction_results_random.json")
    args = parser.parse_args()
    ps.apply()
    r = json.load(open(args.path))
    # the two evaluated frictions from the result file, e.g. labels {"0": "mu_05", "1": "mu_11"} -> ("mu_05", 0.5), ("mu_11", 1.1)
    SURFACES = [(n, int(n[3:]) / 10) for n in r["config"]["labels"].values()]
    cmap = plt.get_cmap(ps.FRICTION_CMAP); norm = mcolors.Normalize(ps.FRICTION_MIN, ps.FRICTION_MAX)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(ps.COLUMN_IN, 1.3))
    pos = [2, 1, 0]
    for ax, (key, mu) in zip(axes, SURFACES):
        data = [[100 * v for v in r[k][key]] for k, _ in ROWS]
        ax.boxplot(data, positions=pos, vert=False, widths=0.6, patch_artist=True, showfliers=False, whis=(10, 90),
                   medianprops=dict(color="black", lw=1.0), boxprops=dict(facecolor=mcolors.to_hex(cmap(norm(mu))), lw=0.6),
                   whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.axvline(np.median(data[1]), color="#444444", lw=0.6, ls=(0, (2, 2)), zorder=0)   # matched median
        ax.set_title(r"$\mu=%.1f$" % mu, fontsize=8, pad=3)
        ax.set_xlim(0, XMAX); ax.set_xticks(range(0, XMAX + 1, 20)); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        print(f"mu {mu}: " + ", ".join(f"{lab} {np.median(d):.1f}" for (_, lab), d in zip(ROWS, data)) + " cm (median)")
    axes[0].set_yticks(pos); axes[0].set_yticklabels([lab for _, lab in ROWS])
    fig.tight_layout(w_pad=0.6, rect=(0, 0.12, 1, 1))
    p0, p1 = axes[0].get_position(), axes[1].get_position()
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.17, "Position error of the prediction, rms over the horizon [cm]", ha="center", va="top",
             fontsize=8)
    ps.save(fig, "evaluation/01_prediction/prediction_error")


if __name__ == "__main__":
    main()

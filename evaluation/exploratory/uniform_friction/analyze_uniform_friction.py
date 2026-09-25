"""Analyze the uniform-friction control runs.

Answers two questions:
  1. Does the implied friction track the true (constant) friction when the
     spatial confound is removed? -> summary table + left panel
  2. Does the discrimination grow with excitation, i.e. is the past window
     informative only near the tire limit? -> right panel, implied friction
     binned by speed

Usage:
    python3 evaluation/exploratory/uniform_friction/analyze_uniform_friction.py
    python3 evaluation/exploratory/uniform_friction/analyze_uniform_friction.py --frictions 0.5 0.8 1.1
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR = os.path.join(REPO_ROOT, "evaluation", "exploratory", "uniform_friction")

DATASET_FRICTIONS = np.array([0.5, 1.1])  # subset labels of the pooled dataset


def load_curvature(map_name):
    """Raceline curvature kappa(s), used to turn speed into lateral acceleration."""
    with open(os.path.join(REPO_ROOT, f"configs/config_{map_name}.yaml")) as f:
        conf = yaml.safe_load(f)
    raceline = np.loadtxt(os.path.join(REPO_ROOT, conf["wpt_path"]),
                          delimiter=conf["wpt_delim"], skiprows=conf["wpt_rowskip"])
    return raceline[:, 0], raceline[:, 4]  # s_m, kappa_radpm


def load_run(map_name, friction, tag=""):
    path = os.path.join(REPO_ROOT, f"log_dpc_eval_{map_name}_uniform_{friction}{tag}")
    with open(path) as f:
        data = json.load(f)

    lap_n = np.asarray(data["lap_n"], dtype=int)
    keep = lap_n >= 1  # discard the first lap, as in the main evaluation

    implied_mu = np.array([
        np.nan if u is None else float(np.asarray(u, dtype=float) @ DATASET_FRICTIONS)
        for u in data["usage_fraction"]
    ])

    progress = np.asarray(data["track_progress"], dtype=float)[keep]
    speed = np.asarray(data["vx"], dtype=float)[keep]

    # Excitation proxy: friction only shapes the dynamics near the tire limit,
    # i.e. at high lateral acceleration, not at high speed (straights are fast).
    s_ref, kappa_ref = load_curvature(map_name)
    kappa = np.interp(progress, s_ref, kappa_ref, period=s_ref[-1])
    lat_acc = np.abs(speed ** 2 * kappa)

    return {
        "friction": friction,
        "implied_mu": implied_mu[keep],
        "progress": progress,
        "speed": speed,
        "lat_acc": lat_acc,
        "tracking_error": np.asarray(data["tracking_error"], dtype=float)[keep],
    }


def summarize(runs):
    neutral = DATASET_FRICTIONS.mean()  # what a non-discriminating selector gives
    print(f"\n{'true mu':>8} {'implied (median)':>18} {'IQR':>14} {'bias vs neutral':>16} {'tracking err':>13}")
    print("-" * 74)
    for run in runs:
        mu_hat = run["implied_mu"]
        mu_hat = mu_hat[~np.isnan(mu_hat)]
        q25, q50, q75 = np.percentile(mu_hat, [25, 50, 75])
        # positive = implied friction moved from the neutral value toward the truth
        direction = np.sign(run["friction"] - neutral)
        bias = direction * (q50 - neutral)
        print(f"{run['friction']:>8.2f} {q50:>18.3f} {f'[{q25:.3f}, {q75:.3f}]':>14} "
              f"{bias:>+16.3f} {np.nanmean(run['tracking_error']):>13.3f}")

    if len(runs) >= 2:
        lo, hi = runs[0], runs[-1]
        span = hi["friction"] - lo["friction"]

        def separation(mask_fn=None):
            vals = []
            for run in (lo, hi):
                mu_hat = run["implied_mu"]
                sel = ~np.isnan(mu_hat)
                if mask_fn is not None:
                    sel &= mask_fn(run)
                vals.append(np.median(mu_hat[sel]) if sel.sum() >= 20 else np.nan)
            return vals[1] - vals[0]

        # Split by excitation: friction is only observable near the tire limit.
        all_lat = np.concatenate([r["lat_acc"] for r in runs])
        lo_thr, hi_thr = np.percentile(all_lat, [50, 80])

        print(f"\n{'regime':>28} {'separation':>12} {'% of true':>11}")
        print("-" * 54)
        for label, fn in [
            ("all samples", None),
            (f"straights (a_lat < {lo_thr:.1f})", lambda r: r["lat_acc"] < lo_thr),
            (f"corners (a_lat > {hi_thr:.1f})", lambda r: r["lat_acc"] > hi_thr),
        ]:
            sep = separation(fn)
            print(f"{label:>28} {sep:>+12.3f} {100 * sep / span:>10.0f}%")
        print("\nThe mechanism predicts a small separation on straights and a large one\n"
              "in corners, where the friction difference is observable.")


def plot(runs, map_name):
    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(11, 3.5))

    for run in runs:
        color = plt.cm.coolwarm((run["friction"] - 0.5) / 0.6)
        label = rf"$\mu={run['friction']}$"

        order = np.argsort(run["progress"])
        ax_p.plot(run["progress"][order], run["implied_mu"][order],
                  ".", markersize=2, alpha=0.3, color=color)
        ax_p.axhline(run["friction"], color=color, linestyle="--", linewidth=1)
        ax_p.plot([], [], color=color, label=label)

        # implied friction vs lateral acceleration: does discrimination need
        # excitation? Speed is the wrong proxy - straights are fast but carry
        # no friction information.
        excitation = run["lat_acc"]
        mu_hat = run["implied_mu"]
        valid = ~np.isnan(mu_hat)
        if valid.sum() > 20:
            bins = np.linspace(np.nanpercentile(excitation[valid], 2),
                               np.nanpercentile(excitation[valid], 98), 9)
            idx = np.digitize(excitation[valid], bins)
            centers, medians = [], []
            for b in range(1, len(bins)):
                sel = idx == b
                if sel.sum() >= 5:
                    centers.append(0.5 * (bins[b - 1] + bins[b]))
                    medians.append(np.median(mu_hat[valid][sel]))
            ax_v.plot(centers, medians, "o-", color=color, label=label)
            ax_v.axhline(run["friction"], color=color, linestyle="--", linewidth=1)

    for ax in (ax_p, ax_v):
        ax.axhline(DATASET_FRICTIONS.mean(), color="gray", linewidth=1,
                   label="neutral (no discrimination)")
        ax.set_ylabel("implied friction")
        ax.legend(fontsize=8)

    ax_p.set_xlabel("track progress [m]")
    ax_p.set_title("Uniform friction: implied vs. true")
    ax_v.set_xlabel(r"lateral acceleration $|v^2\kappa|$ [m/s$^2$]")
    ax_v.set_title("Discrimination vs. excitation")

    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"uniform_friction_{map_name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.01)
    print(f"\nFigure written to evaluation/exploratory/uniform_friction/uniform_friction_{map_name}.pdf")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="l_shape")
    parser.add_argument("--frictions", nargs="+", type=float, default=[0.5, 1.1])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    runs = [load_run(args.map, mu, args.tag) for mu in sorted(args.frictions)]
    summarize(runs)
    plot(runs, args.map + args.tag)


if __name__ == "__main__":
    main()

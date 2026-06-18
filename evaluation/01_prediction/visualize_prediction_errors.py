"""
Plot prediction error results from saved JSON.
Horizontal boxplots, stacked vertically — fits double-column paper layout.

Usage:
    python plot_prediction_results.py results/prediction/prediction_results.json
    python plot_prediction_results.py results/prediction/prediction_results.json --output fig.pdf
"""
import json
import argparse

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors

MU_VALUES = {
    "mu_03": 0.3,
    "mu_07": 0.7,
    "mu_10": 1.0,
}

MU_LABELS = {
    "mu_10": r"$\mu=1.0$",
    "mu_07": r"$\mu=0.7$",
    "mu_03": r"$\mu=0.3$",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)

def set_latex_style(column_width_pt=252.0, fraction=1.0, subplots=(1, 1)):
    inches_per_pt = 1.0 / 72.27

    fig_width_in = column_width_pt * inches_per_pt * fraction
    fig_height_in = fig_width_in * 0.9 * (subplots[0] / subplots[1])

    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.titlesize": 10,
        "figure.figsize": (fig_width_in, fig_height_in),
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    })


def build_rows(results, mu_colors):
    """
    Returns list of (label, errors, color) or None for spacers.
    Order is bottom-to-top.
    Color indicates evaluation friction.
    Labels follow the five settings described in the paper.
    """
    rows = []

    def add_block(label, data):
        names = ["mu_10", "mu_07", "mu_03"]
        for idx, name in enumerate(names):
            row_label = label if idx == 1 else ""
            rows.append((row_label, data[name], mu_colors[name]))

    add_block("(5) Contextual", results["test_5_mixed_contextual"])
    rows.append(None)

    add_block("(4) Random", results["test_4_mixed_random"])
    rows.append(None)

    add_block("(3) Pooled", results["test_3_mixed_full"])
    rows.append(None)

    t2 = results["test_2_cross"]
    rows.append(("", t2["mu_03→mu_10"], mu_colors["mu_10"]))
    #rows.append(("(2) Cross", t2["mu_10→mu_03"], mu_colors["mu_03"]))
    rows.append(("(2) Cross", t2["mu_10→mu_03"], mu_colors["mu_03"]))

    rows.append(None)

    add_block("(1) Self", results["test_1_self"])

    return rows

def plot_results(results, output_path=None):
    #set_latex_style()
    cmap_friction = plt.get_cmap("coolwarm")
    norm_c = mcolors.Normalize(vmin=0.5, vmax=1.1)  # match your other plot exactly

    MU_COLORS = {
        key: mcolors.to_hex(cmap_friction(norm_c(val)))
        for key, val in MU_VALUES.items()
    }

    rows = build_rows(results, MU_COLORS)

    config = results.get("config", {})
    num_traj = config.get("num_traj", "?")

    # Compute y positions with gaps for spacers
    y_positions = []
    labels = []
    colors = []
    data = []
    y = 0
    for r in rows:
        if r is None:
            y += 0.1
        else:
            y_positions.append(y)
            labels.append(r[0])
            data.append(r[1])
            colors.append(r[2])
            y += 0.2

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, (pos, errors, color) in enumerate(zip(y_positions, data, colors)):
        bp = ax.boxplot(
            [errors],
            positions=[pos],
            vert=False,
            #widths=0.3,
            patch_artist=True,
            showfliers=False,
            #flierprops=dict(marker="o", markersize=2.5, markerfacecolor=color, markeredgecolor=color, alpha=0.5),
            medianprops=dict(color="black"),
            boxprops=dict(facecolor=color),
            #whiskerprops=dict(color=color, linewidth=0.8),
            #capprops=dict(color=color, linewidth=0.8),
        )

    #ax.set_xscale('log')
    #ax.set_xlabel("RMSE (log scale)", fontsize=9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("RMSE", fontsize=9)
    ax.set_xlim(0, None)
    ax.tick_params(axis='x', labelsize=8)

    '''
    # Section labels on the right
    sections = [
        (f"Contextual (n={num_traj})", 0, 3),
        (f"Random (n={num_traj})", 3, 6),
        ("Full dataset", 6, 9),
        ("Cross-prediction", 9, 11),
        ("Self-prediction", 11, 14),
    ]
    for section_name, start, end in sections:
        mid_y = (y_positions[start] + y_positions[end - 1]) / 2
        ax.annotate(
            section_name,
            xy=(1.02, mid_y),
            xycoords=("axes fraction", "data"),
            fontsize=7.5,
            fontweight="bold",
            va="center",
            ha="left",
        )
    '''

    # Legend
    legend_patches = [mpatches.Patch(facecolor=c, edgecolor="black", label=MU_LABELS[k])
                      for k, c in MU_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", title="Eval surface",
              fontsize=7, title_fontsize=7.5, framealpha=0.9)

    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")
    plt.show()
    fig.savefig("evaluation/01_prediction/prediction_error.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig("evaluation/01_prediction/prediction_error.svg", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to prediction_results.json")
    parser.add_argument("--output", type=str, default=None, help="Save figure to path (e.g. fig.pdf)")
    args = parser.parse_args()

    results = load_results(args.path)
    plot_results(results, output_path=args.output)

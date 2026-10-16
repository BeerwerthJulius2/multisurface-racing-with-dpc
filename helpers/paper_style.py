"""Figure style shared by the paper figures: IEEE column width, Times-like 8 pt text at print size, one colour and line style
per controller, friction always on coolwarm 0.5 (blue) ... 1.1 (red)."""
import matplotlib
import numpy as np

COLUMN_IN = 3.5   # IEEE two-column \columnwidth
COLORS = {"NMPC": "#222222", "pooled": "#1b7f3b", "oracle": "#e69f00", "inverse": "#7b3294"}
LINES = {"NMPC": "-", "pooled": "-", "oracle": (0, (4, 1.5)), "inverse": (0, (1, 1))}
LABELS = {"NMPC": "NMPC", "pooled": "DPC pooled", "oracle": "DPC matched", "inverse": "DPC mismatched"}   # oracle / inverse arms
FRICTION_CMAP, FRICTION_MIN, FRICTION_MAX = "coolwarm", 0.5, 1.1


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"],   # Times-like TrueType shipped with matplotlib
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "paper",
        "savefig.dpi": 300,
    })


def friction_color(mu):
    from matplotlib import colors
    import matplotlib.pyplot as plt
    return colors.to_hex(plt.get_cmap(FRICTION_CMAP)(colors.Normalize(FRICTION_MIN, FRICTION_MAX)(mu)))


def box_rows(ax, groups, gap=0.1, step=0.2):
    """Horizontal boxplots as in Fig. 1 (evaluation/01_prediction): groups = [(label, [(values, colour), ...]), ...] from top
    to bottom, one box per (values, colour); empty values leave the slot free. Returns the label positions."""
    y, ticks = 0.0, []
    for label, boxes in groups:
        pos = []
        for values, color in boxes:
            if len(values):
                ax.boxplot([values], positions=[-y], vert=False, widths=0.15, patch_artist=True, showfliers=False,
                           medianprops=dict(color="black"), boxprops=dict(facecolor=color))
            pos.append(-y); y += step
        ticks.append(np.mean(pos)); y += gap
    ax.set_yticks(ticks); ax.set_yticklabels([g[0] for g in groups]); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-y + gap + step / 2, step / 2); ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    return ticks


def dot_bars(ax, groups, arms, values, ymax=None):
    """Grouped comparison as in the hardware figure: per group (x) the arms side by side, one dot per run / seed and a bar
    at their mean. values[(group, arm)] = list of per-run values (missing or empty: slot left free). Means above ymax are
    drawn at the top edge with an arrow and their value."""
    n = len(arms); width = 0.8 / n
    for gi, g in enumerate(groups):
        for ai, arm in enumerate(arms):
            v = np.asarray(values.get((g, arm), []), float)
            if not len(v):
                continue
            x = gi + (ai - (n - 1) / 2) * width; m = v.mean()
            if ymax is not None and m > ymax:
                ax.annotate(f"{m:.2g}", xy=(x, ymax), xytext=(x, ymax * 0.84), ha="center", va="top", fontsize=6,
                            color=COLORS[arm], arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS[arm]))
                continue
            ax.plot([x - 0.42 * width, x + 0.42 * width], [m, m], color=COLORS[arm], lw=1.6, solid_capstyle="butt")
            ax.scatter(x + np.linspace(-0.2, 0.2, len(v)) * width if len(v) > 1 else [x], v, s=5, color=COLORS[arm],
                       zorder=3, linewidths=0)
    ax.set_xticks(range(len(groups))); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(groups) - 0.5); ax.grid(axis="y", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
    if ymax is not None:
        ax.set_ylim(0, ymax)


MARKERS = {"pooled": ("o", 34), "oracle": ("o", 22), "inverse": ("o", 22), "NMPC": ("D", 18)}


def dumbbell(ax, rows, values, xmax=None, unit=""):
    """One row per group (top to bottom): a light bar from the matched (oracle) to the mismatched (inverse) value marks what
    the data choice is worth, pooled is the large dot on it, NMPC a diamond. values[(row, arm)] = number (missing: not drawn).
    Values beyond xmax are drawn at the edge as an arrow with the number."""
    ys = -np.arange(len(rows))
    for y, r in zip(ys, rows):
        a, b = values.get((r, "oracle")), values.get((r, "inverse"))
        if a is not None and b is not None:
            ax.plot([a, b], [y, y], color="#d9d9d9", lw=4.5, solid_capstyle="round", zorder=1)
        for arm in ("oracle", "inverse", "NMPC", "pooled"):
            v = values.get((r, arm))
            if v is None:
                continue
            m, s = MARKERS[arm]
            if xmax is not None and v > xmax:
                ax.scatter([xmax], [y], marker=">", s=s + 6, color=COLORS[arm], clip_on=False, zorder=2)   # off the scale
                ax.text(xmax * 0.975, y + 0.28, f"{v:.0f}{unit}", ha="right", va="center", fontsize=6.5, color="#222222")
                continue
            ax.scatter([v], [y], marker=m, s=s, color=COLORS[arm], edgecolors="white", linewidths=0.8, zorder=3 if arm == "pooled" else 2)
    ax.set_yticks(ys); ax.set_yticklabels(rows); ax.tick_params(axis="y", length=0)
    ax.set_ylim(ys[-1] - 0.6, 0.6); ax.grid(axis="x", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    if xmax is not None:
        ax.set_xlim(0, xmax)


def arm_legend(fig, arms, y=1.0):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker=MARKERS[a][0], ms=np.sqrt(MARKERS[a][1]) + 0.5, color=COLORS[a], mec="white", mew=0.6,
                      label=LABELS[a]) for a in arms]
    fig.legend(handles=handles, ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, y), handletextpad=0.2, columnspacing=1.0,
               borderaxespad=0.1)


def save(fig, stem):
    """stem.pdf (for \\includegraphics) and stem.svg (for \\includesvg)."""
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0.01, metadata={"CreationDate": None})   # no date: same
    fig.savefig(stem + ".svg", bbox_inches="tight", pad_inches=0.01, metadata={"Date": None})           # file every run
    print("written", stem + ".pdf/.svg")

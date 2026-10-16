"""Figure style shared by the paper figures: IEEE column width, Times-like 8 pt text at print size, one colour and line style
per controller, friction always on coolwarm 0.5 (blue) ... 1.1 (red)."""
import matplotlib

COLUMN_IN = 3.5   # IEEE two-column \columnwidth
COLORS = {"NMPC": "#222222", "pooled": "#1b7f3b", "oracle": "#e69f00", "inverse": "#7b3294"}
LINES = {"NMPC": "-", "pooled": "-", "oracle": (0, (4, 1.5)), "inverse": (0, (1, 1))}
LABELS = {"NMPC": "NMPC", "pooled": "DPC pooled", "oracle": "DPC oracle", "inverse": "DPC inverse"}
FRICTION_CMAP, FRICTION_MIN, FRICTION_MAX = "coolwarm", 0.5, 1.1


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"],   # Times-like TrueType shipped with matplotlib
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "paper",
        "savefig.dpi": 300,
    })


def save(fig, stem):
    """stem.pdf (for \\includegraphics) and stem.svg (for \\includesvg)."""
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0.01, metadata={"CreationDate": None})   # no date: same
    fig.savefig(stem + ".svg", bbox_inches="tight", pad_inches=0.01, metadata={"Date": None})           # file every run
    print("written", stem + ".pdf/.svg")

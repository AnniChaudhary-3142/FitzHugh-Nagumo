"""
Shared figure style for the AIP Advances revision of
"A Structural-Dynamical Decomposition of Synchrony Loss in Stochastic
FitzHugh-Nagumo Networks".

One font family, one size scheme, one colour-blind-safe palette and one
panel-label helper for all ten figures. Nothing here reads or writes the
public repository in fhn-alzheimers/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # fhn-alzheimers/revision
REPO = HERE.parent                              # fhn-alzheimers
FIGDIR = HERE / "figures"

# committed campaign output
RUNS = REPO / "runs"
CANON_ROBUST = RUNS / "canonical" / "out" / "robustness_full_data.npz"
RANDOM_ROBUST = RUNS / "random" / "out" / "robustness_full_data.npz"
CANON_FIGS = RUNS / "canonical_figs" / "alzheimers_results"
RESULTS_NPZ = CANON_FIGS / "results.npz"
RUN_INFO = CANON_FIGS / "run_info.json"

# the 20-realisation percolation sweep run for the revision. The published
# 2-realisation sweep underestimated f_c, see revision/README.md
PERCOLATION = HERE / "runs" / "percolation20" / "percolation_data.npz"

# full (v, w) traces. Excluded by .gitignore because they run to ~10.7 GB.
# Regenerate with save_trajectories_at in Config, see the main README.
TRAJ_DIR = CANON_FIGS / "trajectories_canonical"

N = 1000
K_NBR = 8
P_REWIRE = 0.1
A_OSC = 0.65
B = 0.80
EPS = 0.06
I_EXT = 0.30
K_COUPLE = 0.35
SIGMA = 0.05
DT = 0.01
T_REC = 400.0
BURN_IN = 100.0
N_STAGES = 10
ABLATION_MAX = 0.85
MASTER_SEED = 20260529
LAYOUT_SEED = 20260529

# --------------------------------------------------------------------------
# palette: Okabe-Ito, colour-blind safe and separable in greyscale once
# paired with the marker and line styles below
# --------------------------------------------------------------------------
C_GLOBAL = "#0072B2"     # blue        R_global
C_COMPONENT = "#D55E00"  # vermillion  R_component
C_PRED = "#444444"       # dark grey   participation-ratio bound
C_STRUCT = "#009E73"     # bluish green  LCC / N
C_EFF = "#CC79A7"        # reddish purple  efficiency
C_NEUTRAL = "#333333"
C_ACCENT = "#E69F00"     # orange      single-neuron trajectory

STYLE_GLOBAL = dict(color=C_GLOBAL, marker="o", linestyle="-")
STYLE_COMPONENT = dict(color=C_COMPONENT, marker="s", linestyle="--")

#: stage colours, viridis is colour-blind safe and monotone in luminance
def stage_colour(i: int, n: int) -> tuple:
    return plt.cm.viridis(i / max(n - 1, 1))


# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------
BASE = 10.0
FIG_W = 7.1          # inches, close to the \textwidth of the article class

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": BASE,
    "axes.titlesize": BASE,
    "axes.labelsize": BASE,
    "xtick.labelsize": BASE - 1.0,
    "ytick.labelsize": BASE - 1.0,
    "legend.fontsize": BASE - 1.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.borderpad": 0.4,
    "legend.handlelength": 2.2,
    "figure.titlesize": BASE + 1,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.prop_cycle": plt.cycler(color=[C_GLOBAL, C_COMPONENT, C_STRUCT,
                                         C_ACCENT, C_EFF, C_PRED]),
}


def apply_style() -> None:
    plt.rcParams.update(RC)


# --------------------------------------------------------------------------
# the one panel-label helper used by every figure
# --------------------------------------------------------------------------
PANEL_KW = dict(
    x=0.018, y=0.982,
    fontsize=BASE + 2.0,
    fontweight="bold",
    ha="left", va="top",
)


def panel_label(ax, letter: str, *, dx: float = 0.0, dy: float = 0.0,
                box: bool = True) -> None:
    """Bold lowercase letter in parentheses, inside the panel, top-left.

    Same corner, same size and same weight on every panel of every figure.
    ``dx``/``dy`` are small nudges in axes coordinates for panels whose
    top-left corner is occupied, and ``box`` can be turned off on panels
    with no axes frame.
    """
    bbox = dict(facecolor="white", edgecolor="none", alpha=0.80,
                boxstyle="square,pad=0.16") if box else None
    ax.text(PANEL_KW["x"] + dx, PANEL_KW["y"] + dy, f"({letter})",
            transform=ax.transAxes,
            fontsize=PANEL_KW["fontsize"], fontweight=PANEL_KW["fontweight"],
            ha=PANEL_KW["ha"], va=PANEL_KW["va"], bbox=bbox, zorder=50)


def headroom(ax, frac: float = 0.20) -> None:
    """Grow the top of the y-axis so the panel label never sits on data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def annotate(ax, text: str, *, loc: str = "upper right", fontsize=None):
    """Small in-panel note, for example the stage a panel shows."""
    pos = {"upper right": (0.985, 0.982, "right", "top"),
           "upper left": (0.018, 0.982, "left", "top"),
           "lower right": (0.985, 0.025, "right", "bottom"),
           "lower left": (0.018, 0.025, "left", "bottom")}[loc]
    ax.text(pos[0], pos[1], text, transform=ax.transAxes,
            fontsize=fontsize or (BASE - 0.5), ha=pos[2], va=pos[3],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80,
                      boxstyle="square,pad=0.2"), zorder=50)


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------
def save(fig, stem: str) -> None:
    """Write both PNG at 300 dpi and PDF into figures/."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {stem}.png / {stem}.pdf")


def new_fig(w: float, h: float):
    return plt.subplots(figsize=(w, h))

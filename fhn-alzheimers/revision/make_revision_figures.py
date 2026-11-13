"""
Figures for the new revision results, reviewer points R1-2 and R1-3.

Reads Submission_Alpha/Simulations/runs/ only and simulates nothing. Reuses
figstyle.py so these match the ten rebuilt figures in font, palette, panel
labelling and output convention.

  fig11  network size and average degree          (R1-2)
  fig12  critical fraction and centrality spread  (R1-2)
  fig13  node-based against edge-based attack     (R1-3)
  figS2  within-component order against fragment size (supplementary)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle as S

plt.rcParams.update(S.RC)

SIMS = S.HERE / "runs"   # output of the sim01..sim05 campaigns
SIZES = (250, 500, 1000, 2000, 4000)
DEGREES = (4, 6, 8, 12)

# stage schedule of the published protocol, 0 to 85% of edges in ten stages
STAGE_F = np.array([i / 9.0 * 0.85 for i in range(10)])


def _summary(sim: str, cond: str) -> dict:
    return json.loads((SIMS / sim / cond / "summary.json").read_text())


def _seq_colour(i: int, n: int):
    return plt.cm.viridis(0.08 + 0.82 * i / max(n - 1, 1))


# ==========================================================================
# Figure 11 — network size and average degree
# ==========================================================================
def fig11():
    fig, axgrid = plt.subplots(2, 2, figsize=(S.FIG_W, 3.95))
    plt.subplots_adjust(left=0.082, right=0.995, top=0.945, bottom=0.095,
                        wspace=0.22, hspace=0.42)
    # order: (a) R_g vs N, (b) R_c vs N, (c) R_g vs <k>, (d) R_c vs <k>
    axes = [axgrid[0, 0], axgrid[0, 1], axgrid[1, 0], axgrid[1, 1]]

    arms = [("sim01_size_sweep", [f"N{n}" for n in SIZES],
             [f"$N={n}$" for n in SIZES], "N"),
            ("sim02_degree_sweep", [f"k{k}" for k in DEGREES],
             [rf"$\langle k\rangle={k}$" for k in DEGREES], "k")]

    for col, (sim, conds, labels, _) in enumerate(arms):
        for which, key in enumerate(("R_global_mean", "R_component_mean")):
            ax = axes[col * 2 + which]
            for i, (c, lab) in enumerate(zip(conds, labels)):
                s = _summary(sim, c)
                ax.plot(STAGE_F, s[key], marker="o", ms=2.6, lw=1.3,
                        color=_seq_colour(i, len(conds)), label=lab)
            if col == 1:
                ax.set_xlabel("Edge removal fraction $f$")
            ax.set_xlim(-0.03, 0.88)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.grid(alpha=0.25, lw=0.5)
            if which == 0:
                ax.set_ylim(0.28, 1.01); ax.set_ylabel("$R_g$")
            else:
                ax.set_ylim(0.90, 1.005); ax.set_ylabel("$R_c$")

    for ax, L in zip(axes, "abcd"):
        S.panel_label(ax, L)
    axes[0].legend(fontsize=S.BASE - 3.2, frameon=False, loc="lower left",
                   handlelength=1.0, borderpad=0.1, labelspacing=0.18)
    axes[2].legend(fontsize=S.BASE - 3.2, frameon=False, loc="lower left",
                   handlelength=1.0, borderpad=0.1, labelspacing=0.18)
    axes[0].set_title("network size, $\\langle k\\rangle = 8$",
                      fontsize=S.BASE - 1.5, pad=3)
    axes[1].set_title("network size, $\\langle k\\rangle = 8$",
                      fontsize=S.BASE - 1.5, pad=3)
    axes[2].set_title("average degree, $N = 1000$",
                      fontsize=S.BASE - 1.5, pad=3)
    axes[3].set_title("average degree, $N = 1000$",
                      fontsize=S.BASE - 1.5, pad=3)
    S.save(fig, "fig11_size_degree")


# ==========================================================================
# Figure 12 — critical fraction and the betweenness distribution
# ==========================================================================
def fig12():
    perc = json.loads((SIMS / "sim04_percolation_scaling"
                       / "percolation_scaling.json").read_text())
    cent = json.loads((SIMS / "sim03_centrality_stats"
                       / "centrality_stats.json").read_text())

    def fc(arm, N=None, k=None):
        rows = [r for r in perc if r["arm"] == arm
                and (N is None or r["N"] == N) and (k is None or r["k_nbr"] == k)]
        out = []
        for r in rows:
            f = np.asarray(r["fracs"]); rel = np.asarray(r["lcc_frac_mean"])
            a = np.nonzero(rel >= 0.5)[0]; b = np.nonzero(rel < 0.5)[0]
            v = None
            if a.size and b.size and b[0] - a[-1] == 1:
                i1, i2 = a[-1], b[0]
                v = f[i1] + (f[i2] - f[i1]) * (0.5 - rel[i1]) / (rel[i2] - rel[i1])
            out.append((r["N"], r["k_nbr"], v))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W, 2.15))
    plt.subplots_adjust(left=0.072, right=0.995, top=0.95, bottom=0.225,
                        wspace=0.42)

    # (a) f_c against N, targeted and random
    ax = axes[0]
    for arm, lab, col, mk in (("edge_targeted", "targeted", S.C_GLOBAL, "o"),
                              ("edge_random", "random", S.C_COMPONENT, "s")):
        pts = sorted((n, v) for n, k, v in fc(arm, k=8) if v is not None)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=3.4,
                lw=1.4, color=col, label=lab)
    ax.set_xscale("log"); ax.set_xticks(list(SIZES))
    ax.set_xticklabels([str(n) for n in SIZES], fontsize=S.BASE - 2.6)
    ax.set_xlabel("Network size $N$"); ax.set_ylabel("$f_c$")
    ax.set_ylim(0, 0.95); ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=S.BASE - 3, frameon=False, loc="center left",
              handlelength=1.1, borderpad=0.1)

    # (b) f_c against average degree
    ax = axes[1]
    for arm, lab, col, mk in (("edge_targeted", "targeted", S.C_GLOBAL, "o"),
                              ("edge_random", "random", S.C_COMPONENT, "s")):
        pts = sorted((k, v) for n, k, v in fc(arm, N=1000) if v is not None)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=3.4,
                lw=1.4, color=col, label=lab)
    ax.set_xticks(list(DEGREES))
    ax.set_xlabel(r"Average degree $\langle k\rangle$")
    ax.set_ylabel("$f_c$"); ax.set_ylim(0, 0.95); ax.grid(alpha=0.25, lw=0.5)

    # (c) betweenness concentration against N
    ax = axes[2]
    rows = [c for c in cent if c["arm"] == "size"]
    rows.sort(key=lambda c: c["N"])
    Ns = [c["N"] for c in rows]
    ax.plot(Ns, [c["edge_betweenness"]["gini"] for c in rows], marker="o",
            ms=3.4, lw=1.4, color=S.C_STRUCT, label="Gini")
    ax.plot(Ns, [c["edge_betweenness"]["top10pct_share"] for c in rows],
            marker="^", ms=3.4, lw=1.4, color=S.C_ACCENT, label="top 10% share")
    ax.set_xscale("log"); ax.set_xticks(list(SIZES))
    ax.set_xticklabels([str(n) for n in SIZES], fontsize=S.BASE - 2.6)
    ax.set_xlabel("Network size $N$")
    ax.set_ylabel("Concentration")
    ax.set_ylim(0.30, 0.56); ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=S.BASE - 3, frameon=False, loc="center left",
              handlelength=1.1, borderpad=0.1)

    for ax, L in zip(axes, "abc"):
        S.panel_label(ax, L)
    S.save(fig, "fig12_fc_scaling")


# ==========================================================================
# Figure 13 — node-based against edge-based attack
# ==========================================================================
PROTO = [("edge_targeted",         "edge, targeted",   S.C_GLOBAL,    "-",  "o"),
         ("edge_random",           "edge, random",     S.C_GLOBAL,    "--", "s"),
         ("node_targeted_isolate", "node, targeted",   S.C_COMPONENT, "-",  "o"),
         ("node_random_isolate",   "node, random",     S.C_COMPONENT, "--", "s"),
         ("node_targeted_excise",  "node, excised",    S.C_PRED,      "-",  "^"),
         ("node_random_excise",    "node, excised, random", S.C_PRED, "--", "v")]


def fig13():
    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, 2.50))
    plt.subplots_adjust(left=0.085, right=0.995, top=0.95, bottom=0.185,
                        wspace=0.24)
    for key, ax, ylab, ylim in ((("R_global_mean"), axes[0], "$R_g$", (0.28, 1.02)),
                                (("R_component_mean"), axes[1], "$R_c$", (0.90, 1.005))):
        for proto, lab, col, ls, mk in PROTO:
            s = _summary("sim05_node_ablation", proto)
            x = s["edge_fraction_achieved_mean"]
            ax.plot(x, s[key], marker=mk, ms=2.8, lw=1.3, color=col,
                    linestyle=ls, label=lab)
        ax.set_xlabel("Edge removal fraction actually achieved")
        ax.set_ylabel(ylab); ax.set_ylim(*ylim); ax.set_xlim(-0.03, 0.88)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=S.BASE - 3.2, frameon=False, loc="lower left",
                   handlelength=1.6, borderpad=0.1, labelspacing=0.2)
    for ax, L in zip(axes, "ab"):
        S.panel_label(ax, L)
    S.save(fig, "fig13_node_vs_edge")


# ==========================================================================
# Supplementary — R_c against surviving fragment size
# ==========================================================================
def figS2():
    fig, ax = plt.subplots(figsize=(S.FIG_W * 0.56, 2.25))
    plt.subplots_adjust(left=0.165, right=0.975, top=0.96, bottom=0.185)
    xs, ys, labs = [], [], []
    for k in DEGREES:
        m = json.loads((SIMS / "sim02_degree_sweep" / f"k{k}"
                        / "metrics_per_network.json").read_text())
        s = _summary("sim02_degree_sweep", f"k{k}")
        last = [net[-1] for net in m]
        iso = np.mean([d["n_isolates"] for d in last])
        nc = np.mean([d["n_components"] for d in last])
        xs.append((1000 - iso) / max(nc - iso, 1))
        ys.append(s["R_component_mean"][-1]); labs.append(k)
    ax.plot(xs, ys, marker="o", ms=4.2, lw=1.4, color=S.C_COMPONENT)
    for x, y, k in zip(xs, ys, labs):
        ax.annotate(rf"$\langle k\rangle={k}$", (x, y), fontsize=S.BASE - 2.8,
                    textcoords="offset points", xytext=(5, 4))
    ax.set_xlabel("Mean surviving fragment size (neurons)")
    ax.set_ylabel("$R_c$ at Stage 10")
    ax.grid(alpha=0.25, lw=0.5); ax.set_xlim(0, 13.5)
    S.save(fig, "figS2_fragment_size")


if __name__ == "__main__":
    for fn in (fig11, fig12, fig13, figS2):
        print(f"{fn.__name__} ...")
        fn()
    print("done ->", S.FIGDIR)

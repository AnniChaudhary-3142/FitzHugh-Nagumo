"""
Rebuilds the ten manuscript figures for the AIP Advances revision.

Every panel is drawn by a function of one (or, for the raster figure, two)
matplotlib axes. Each such function is called twice, once into the assembled
multi-panel figure the paper uses, so the
authors can rearrange panels later without re-deriving anything.

Panel letters are drawn inside the panels by figstyle.panel_label, the one
shared helper, so the corner, size and weight are identical everywhere.

Usage
-----
    python3 make_figures.py            # all ten
    python3 make_figures.py 4 7        # only figures 4 and 7
"""
from __future__ import annotations

import sys

import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import figstyle as S
import fhndata as D

S.apply_style()

STAGES_DIAG = (1, 3, 5, 7, 10)
STAGES_SNAP = (1, 5, 10)
LETTERS = "abcdefgh"


# ==========================================================================
# panel plumbing
# ==========================================================================


# ==========================================================================
# Figure 1 — single-neuron phase portrait
# ==========================================================================
def _isolated_neuron():
    """Reproduces fhn_alzheimers.plot_phase_portrait, same seed, same noise."""
    rng = np.random.default_rng(42)
    v_star, w_star = D.fhn_fixed_point()
    v, w = v_star + 0.05, w_star
    n = int(200.0 / S.DT)
    V = np.empty(n)
    W = np.empty(n)
    sdt = np.sqrt(S.DT)
    for i in range(n):
        dv = v - v ** 3 / 3.0 - w + S.I_EXT
        dw = S.EPS * (v + S.A_OSC - S.B * w)
        v += S.DT * dv + S.SIGMA * sdt * rng.standard_normal()
        w += S.DT * dw
        V[i] = v
        W[i] = w
    return V, W, v_star, w_star


def fig01():
    V, W, v_star, w_star = _isolated_neuron()
    t = np.arange(len(V)) * S.DT

    def pa(ax):
        ax.plot(t, V, lw=0.7, color=S.C_GLOBAL)
        ax.set_xlabel("Time (model time units)")
        ax.set_ylabel("Membrane variable  $v(t)$")
        ax.set_xlim(0, 200)
        S.headroom(ax, 0.22)
        S.panel_label(ax, "a")

    def pb(ax):
        vr = np.linspace(-2.45, 2.45, 400)
        ax.plot(V[2000:], W[2000:], lw=0.7, color=S.C_ACCENT, alpha=0.95,
                zorder=2)
        ax.plot(vr, vr - vr ** 3 / 3.0 + S.I_EXT, "--", color=S.C_COMPONENT,
                lw=1.5, zorder=3)
        ax.plot(vr, (vr + S.A_OSC) / S.B, "-.", color=S.C_GLOBAL, lw=1.5,
                zorder=3)
        ax.plot(v_star, w_star, "o", color="k", ms=6.5, zorder=5)
        lab = dict(fontsize=S.BASE - 1.5,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                             boxstyle="square,pad=0.14"), zorder=6)
        ax.text(-1.75, 2.35, r"$v$-nullcline", color=S.C_COMPONENT, **lab)
        ax.text(1.55, 2.35, r"$w$-nullcline", color=S.C_GLOBAL, ha="right",
                **lab)
        ax.text(-0.95, 1.45, "limit cycle", color="#9A6B00", ha="center",
                **lab)
        ax.annotate(f"unstable fixed point\n$({v_star:.3f},\\ {w_star:.3f})$",
                    xy=(v_star, w_star), xytext=(-2.35, -1.25),
                    fontsize=S.BASE - 2.0, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"),
                    zorder=6)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-1.7, 3.0)
        ax.set_xlabel("Membrane variable  $v$")
        ax.set_ylabel("Recovery variable  $w$")
        S.panel_label(ax, "b")

    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, 2.9))
    pa(axes[0])
    pb(axes[1])
    fig.tight_layout()
    S.save(fig, "fig01_phase_portrait")


# ==========================================================================
# Figures 2 and 3 — ensemble R curves, targeted and random ablation
# ==========================================================================
def _R_panels(npz_path):
    d = np.load(npz_path)
    Rg, Rc = d["R_global"], d["R_component"]           # (n_net, n_stage, n_trial)
    n_net, n_stage, n_trial = Rg.shape
    stages = np.arange(1, n_stage + 1)
    g = Rg.mean(axis=2)
    c = Rc.mean(axis=2)
    g_mu, g_sd = g.mean(0), g.std(0, ddof=1)
    c_mu, c_sd = c.mean(0), c.std(0, ddof=1)
    g_se, c_se = g_sd / np.sqrt(n_net), c_sd / np.sqrt(n_net)

    def pa(ax):
        for i in range(n_net):
            ax.plot(stages, g[i], "-", color=S.C_GLOBAL, alpha=0.30, lw=0.7,
                    zorder=1)
            ax.plot(stages, c[i], "-", color=S.C_COMPONENT, alpha=0.30, lw=0.7,
                    zorder=1)
        ax.errorbar(stages, g_mu, yerr=g_sd, **S.STYLE_GLOBAL, lw=2.0, ms=5.5,
                    capsize=3, zorder=3,
                    label=rf"$R_{{\rm g}}$  mean $\pm$ SD ($n={n_net}$ networks)")
        ax.errorbar(stages, c_mu, yerr=c_sd, **S.STYLE_COMPONENT, lw=2.0,
                    ms=5.0, capsize=3, zorder=3,
                    label=r"$R_{\rm c}$  mean $\pm$ SD")
        ax.set_xlabel("Disease stage")
        ax.set_ylabel("Kuramoto order parameter  $R$")
        ax.set_xticks(stages)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylim(0, 1.18)
        ax.legend(loc="lower left", fontsize=S.BASE - 2.0)
        S.panel_label(ax, "a")

    def pb(ax):
        ax.fill_between(stages, g_mu - g_sd, g_mu + g_sd, alpha=0.22,
                        color=S.C_GLOBAL, lw=0, zorder=1,
                        label=r"$\pm 1$ SD across networks")
        ax.fill_between(stages, c_mu - c_sd, c_mu + c_sd, alpha=0.22,
                        color=S.C_COMPONENT, lw=0, zorder=1)
        ax.errorbar(stages, g_mu, yerr=g_se, **S.STYLE_GLOBAL, lw=2.0, ms=5.5,
                    capsize=3, zorder=3,
                    label=r"$R_{\rm g}$  mean $\pm$ SE")
        ax.errorbar(stages, c_mu, yerr=c_se, **S.STYLE_COMPONENT, lw=2.0,
                    ms=5.0, capsize=3, zorder=3,
                    label=r"$R_{\rm c}$  mean $\pm$ SE")
        ax.set_xlabel("Disease stage")
        ax.set_ylabel("Kuramoto order parameter  $R$")
        ax.set_xticks(stages)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylim(0, 1.18)
        ax.legend(loc="lower left", fontsize=S.BASE - 2.0)
        S.panel_label(ax, "b")

    return pa, pb, (g_mu, g_sd, c_mu, c_sd, n_net, n_trial)


def _R_figure(stem, npz_path):
    pa, pb, stats = _R_panels(npz_path)
    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, 2.55))
    pa(axes[0])
    pb(axes[1])
    fig.tight_layout(w_pad=1.2)
    S.save(fig, stem)
    return stats


def fig02():
    return _R_figure("fig02_R_multi_network_robustness", S.CANON_ROBUST)


def fig03():
    return _R_figure("fig03_R_multi_network_random", S.RANDOM_ROBUST)


# ==========================================================================
# Figure 4 — stage-resolved structural metrics
# ==========================================================================
GM_KEYS = [
    ("n_edges", "Number of edges  $|E|$"),
    ("largest_cc_size", "Largest component\nsize (nodes)"),
    ("global_efficiency", "Global efficiency\n$E_{\\rm glob}$"),
    ("mean_clustering", "Mean clustering\ncoefficient  $C$"),
    ("modularity", "Modularity  $Q$ (greedy)"),
    ("avg_path_length_lcc", "Mean path length\nin LCC (edges)"),
]


def fig04():
    metrics = D.stage_metrics()
    stages = np.arange(1, len(metrics) + 1)

    def make(key, ylabel, letter, mark=False, show_x=True):
        vals = np.array([m[key] for m in metrics], dtype=float)

        def draw(ax):
            ax.axvline(4.5, color="0.45", ls=(0, (4, 3)), lw=0.9, zorder=1)
            ax.plot(stages, vals, "o-", color=S.C_GLOBAL, lw=1.8, ms=4.5,
                    zorder=3)
            if show_x:
                ax.set_xlabel("Disease stage")
            ax.set_ylabel(ylabel)
            ax.set_xticks(stages)
            ax.tick_params(axis="x", labelsize=S.BASE - 2.0)
            ax.margins(y=0.10)
            S.headroom(ax, 0.14)
            if mark:
                ax.text(5.05, 0.60, "percolation\nbetween\nstages 4 and 5",
                        transform=ax.get_xaxis_transform(), fontsize=S.BASE - 2.5,
                        ha="left", va="center", color="0.25", linespacing=1.25,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.8, boxstyle="square,pad=0.16"),
                        zorder=50)
            S.panel_label(ax, letter)
        return draw

    grid = [make(k, lab, LETTERS[i], mark=(i == 1), show_x=(i >= 3))
            for i, (k, lab) in enumerate(GM_KEYS)]
    solo = [make(k, lab, LETTERS[i], mark=(i == 1))
            for i, (k, lab) in enumerate(GM_KEYS)]

    fig, axes = plt.subplots(2, 3, figsize=(S.FIG_W, 3.85))
    for ax, draw in zip(axes.flat, grid):
        draw(ax)
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    fig.align_ylabels()
    S.save(fig, "fig04_graph_metrics")


# ==========================================================================
# Figure 5 — spike rasters and mean field at Stages 1, 5, 10
# ==========================================================================
def fig05():
    r = D.raster_data(STAGES_SNAP)
    metrics = D.stage_metrics()

    def make(stage, letter):
        nidx = r[f"nidx_{stage}"]
        tspk = r[f"tspk_{stage}"]
        t = r[f"t_{stage}"]
        mv = r[f"mv_{stage}"]
        m = metrics[stage - 1]
        frac = D.ablation_fraction(stage)
        plural = "s" if m["n_components"] != 1 else ""
        full = (f"Stage {stage}   $f={frac:.2f}$   "
                f"{m['n_components']} component{plural}   "
                f"LCC $=$ {m['largest_cc_size']}")
        short = f"Stage {stage}   $f = {frac:.2f}$"

        def draw(ax_top, ax_bot, compact=False, show_y=True, show_x=True):
            ax_top.scatter(tspk, nidx, s=1.6, c="k", marker="|",
                           linewidths=0.45, rasterized=True)
            ax_top.set_ylim(-6, 238)
            ax_top.set_xlim(t[0], t[-1])
            ax_top.set_yticks([0, 100, 200])
            ax_top.grid(False)
            ax_top.tick_params(labelbottom=False)
            if show_y:
                ax_top.set_ylabel("Neuron index\n(first 200)")
            else:
                ax_top.tick_params(labelleft=False)
            ax_top.text(0.975, 0.96, short if compact else full,
                        transform=ax_top.transAxes, ha="right", va="top",
                        fontsize=S.BASE - 2.0,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, boxstyle="square,pad=0.2"),
                        zorder=50)
            S.panel_label(ax_top, letter, dy=-0.022)

            ax_bot.plot(t, mv, lw=0.9, color=S.C_GLOBAL)
            ax_bot.set_xlim(t[0], t[-1])
            ax_bot.set_ylim(-2.45, 2.45)
            ax_bot.set_yticks([-2, 0, 2])
            ax_bot.set_xticks([0, 50, 100, 150])
            if show_y:
                ax_bot.set_ylabel(r"$\langle v \rangle (t)$")
            else:
                ax_bot.tick_params(labelleft=False)
            if show_x:
                ax_bot.set_xlabel("Time (model time units)")
        return draw

    draws = [(make(st, LETTERS[i]), LETTERS[i])
             for i, st in enumerate(STAGES_SNAP)]

    def _pair(fig, outer, hspace=0.07):
        inner = outer.subgridspec(2, 1, height_ratios=[3.0, 1.15],
                                  hspace=hspace)
        a1 = fig.add_subplot(inner[0])
        a2 = fig.add_subplot(inner[1], sharex=a1)
        return a1, a2

    # assembled: one row of three columns, matching the submitted arrangement
    fig = plt.figure(figsize=(S.FIG_W, 2.02))
    gs = fig.add_gridspec(1, 3, wspace=0.17, left=0.098, right=0.995,
                          top=0.975, bottom=0.215)
    for i, (draw, _) in enumerate(draws):
        a1, a2 = _pair(fig, gs[i], hspace=0.10)
        draw(a1, a2, compact=True, show_y=(i == 0), show_x=False)
    fig.supxlabel("Time (model time units)", fontsize=S.BASE, y=0.015)
    S.save(fig, "fig05_rasters")


# ==========================================================================
# Figure 6 — network visualisations at Stages 1, 5, 10
# ==========================================================================
COMP_COLOURS = ["#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
                "#F0E442", "#8C8C8C", "#117733", "#882255", "#44AA99"]


def fig06():
    lay = D.network_layouts(STAGES_SNAP)
    metrics = D.stage_metrics()

    def make(stage, letter):
        pos = lay[f"pos_{stage}"]
        nodes = lay[f"nodes_{stage}"]
        edges = lay[f"edges_{stage}"]
        comp = lay[f"comp_{stage}"]
        index = {int(n): i for i, n in enumerate(nodes)}
        m = metrics[stage - 1]

        cols = []
        for n in nodes:
            c = comp[n]
            cols.append(S.C_GLOBAL if c == 0
                        else COMP_COLOURS[(c - 1) % len(COMP_COLOURS)])

        segs = np.array([[pos[index[int(u)]], pos[index[int(v)]]]
                         for u, v in edges]) if len(edges) else np.empty((0, 2, 2))

        note = (f"Stage {stage}\n$f = {D.ablation_fraction(stage):.2f}$\n"
                f"$|E| = {m['n_edges']}$\n"
                f"{m['n_components']} component"
                f"{'s' if m['n_components'] != 1 else ''}\n"
                f"LCC $=$ {m['largest_cc_size']}")

        def draw(ax):
            if len(segs):
                ax.add_collection(LineCollection(segs, colors="0.55",
                                                 linewidths=0.25, alpha=0.5,
                                                 zorder=1, rasterized=True))
            ax.scatter(pos[:, 0], pos[:, 1], s=4.0, c=cols, linewidths=0,
                       zorder=2, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.7)
                sp.set_color("0.7")
            ax.set_aspect("equal")
            ax.autoscale_view()
            lo = pos.min(0)
            hi = pos.max(0)
            pad = 0.06 * (hi - lo).max()
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad * 3.2)
            ax.text(0.98, 0.98, note, transform=ax.transAxes, ha="right",
                    va="top", fontsize=S.BASE - 2.5, linespacing=1.25,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                              boxstyle="square,pad=0.22"), zorder=50)
            S.panel_label(ax, letter)
        return draw

    draws = [(make(s, LETTERS[i]), LETTERS[i])
             for i, s in enumerate(STAGES_SNAP)]

    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W, 2.55))
    for ax, (draw, _) in zip(axes, draws):
        draw(ax)
    fig.tight_layout(w_pad=0.6)
    S.save(fig, "fig06_network_viz")


# ==========================================================================
# Figure 7 — phase-difference distributions
# ==========================================================================
def fig07():
    pd = D.phase_distribution_data(STAGES_SNAP)
    bins = np.linspace(-np.pi, np.pi, 61)
    xt = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtl = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

    def finish(ax, title, note):
        ax.axhline(1.0 / (2 * np.pi), color="k", ls=":", lw=1.1, zorder=4)
        ax.text(np.pi * 0.96, 1.0 / (2 * np.pi), r"$1/2\pi$", ha="right",
                va="bottom", fontsize=S.BASE - 3.0, zorder=5)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)
        ax.set_ylabel("Density (rad$^{-1}$)")
        ax.set_title(title, fontsize=S.BASE - 1.0, pad=4)
        S.headroom(ax, 0.45)
        ax.text(0.97, 0.965, note, transform=ax.transAxes, ha="right",
                va="top", fontsize=S.BASE - 2.8, linespacing=1.3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          boxstyle="square,pad=0.18"), zorder=50)

    def make_within(stage, letter, compact=False):
        x = pd[f"within_{stage}"]
        fw = float(pd[f"fwhm_{stage}"])
        ncomp = int(pd[f"ncomp_{stage}"])

        def draw(ax):
            ax.hist(x, bins=bins, density=True, color=S.C_GLOBAL, alpha=0.9,
                    edgecolor="none")
            ax.set_xlabel(("Phase deviation (rad)" if compact else
                           "Phase deviation from\ncomponent mean (rad)"))
            finish(ax, f"Stage {stage}, within component",
                   f"{ncomp} component{'s' if ncomp != 1 else ''}\n"
                   f"of size $\\geq 2$\nFWHM $=$ {fw:.2f} rad")
            S.panel_label(ax, letter)
        return draw

    def make_between(stage, letter, compact=False):
        x = pd[f"pair_d_{stage}"]
        w = pd[f"pair_w_{stage}"]
        ncomp = int(pd[f"ncomp_{stage}"])

        def draw(ax):
            ax.hist(x, bins=bins, weights=w, density=True,
                    color=S.C_COMPONENT, alpha=0.9, edgecolor="none")
            ax.set_xlabel(("Mean-phase difference (rad)" if compact else
                           "Difference of component\nmean phases (rad)"))
            finish(ax, f"Stage {stage}, between components",
                   f"{ncomp} components\npairs weighted by\nthe product of sizes")
            S.panel_label(ax, letter)
        return draw

    between = [st for st in STAGES_SNAP if f"pair_d_{st}" in pd]

    def build(compact):
        out = [(make_within(st, LETTERS[i], compact), LETTERS[i])
               for i, st in enumerate(STAGES_SNAP)]
        out += [(make_between(st, LETTERS[3 + j], compact), LETTERS[3 + j])
                for j, st in enumerate(between)]
        return out

    fig = plt.figure(figsize=(S.FIG_W, 3.80))
    gs = fig.add_gridspec(2, 6, hspace=0.58, wspace=1.40, left=0.085,
                          right=0.985, top=0.935, bottom=0.115)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    for (draw, _), slot in zip(build(True), slots):
        draw(fig.add_subplot(slot))
    S.save(fig, "fig07_phase_distributions")


# ==========================================================================
# Figure 8 — mean-field power spectrum
# ==========================================================================
def fig08():
    sp = D.spectra_data(STAGES_DIAG)
    stages = list(sp["stages"])
    f_peak = sp["f_peak"]
    psd_peak = sp["psd_peak"]
    psd_tot = sp["psd_total"]

    def pa(ax):
        for i, s in enumerate(stages):
            ax.semilogy(sp[f"f_{s}"], sp[f"psd_{s}"],
                        color=S.stage_colour(i, len(stages)), lw=1.5,
                        label=f"Stage {s}")
        ax.axvline(float(f_peak[0]), color="0.35", ls=":", lw=1.1, zorder=1)
        ax.annotate(f"$f_{{\\rm peak}} = {f_peak[0]:.4f}$\n"
                    f"period $= {1/f_peak[0]:.0f}$ time units\n"
                    "same at every stage",
                    xy=(float(f_peak[0]), float(psd_peak[0])),
                    xytext=(0.26, 0.52), textcoords="axes fraction",
                    fontsize=S.BASE - 2.2, linespacing=1.3,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                              boxstyle="square,pad=0.18"),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35"))
        ax.set_xlim(0, 0.4)
        ax.set_ylim(1e-7, 1e5)
        ax.set_xlabel("Frequency  $f$  (1 / time unit)")
        ax.set_ylabel(r"PSD of $\langle v \rangle (t)$"
                      "\n(power per unit frequency)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(loc="upper right", ncol=2, fontsize=S.BASE - 2.5)
        S.panel_label(ax, "a")

    def pb(ax):
        ax.semilogy(stages, psd_peak, "o-", color=S.C_GLOBAL, lw=1.8, ms=5.5,
                    label="peak PSD at $f_{\\rm peak}$")
        ax.semilogy(stages, psd_tot, "s--", color=S.C_COMPONENT, lw=1.8,
                    ms=5.0, label="integrated PSD (total power)")
        ax.set_xlabel("Disease stage")
        ax.set_ylabel("Mean-field power (log scale)")
        ax.set_xticks(stages)
        ax.set_ylim(1e-1, 1e5)
        ax.legend(loc="lower left", fontsize=S.BASE - 2.0)
        ax.text(0.97, 0.47,
                f"peak  {psd_peak[0]:.1f} $\\rightarrow$ {psd_peak[-1]:.1f}"
                f"  ({psd_peak[0]/psd_peak[-1]:.1f}$\\times$)\n"
                f"total  {psd_tot[0]:.2f} $\\rightarrow$ {psd_tot[-1]:.2f}"
                f"  ({psd_tot[0]/psd_tot[-1]:.1f}$\\times$)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=S.BASE - 2.0, linespacing=1.3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          boxstyle="square,pad=0.2"), zorder=50)
        S.panel_label(ax, "b")

    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, 3.1))
    pa(axes[0])
    pb(axes[1])
    fig.tight_layout()
    S.save(fig, "fig08_mean_field_spectra")

    return dict(f_peak=f_peak, psd_peak=psd_peak, psd_total=psd_tot)


# ==========================================================================
# Figure 9 — periodicity diagnostics
# ==========================================================================
def fig09():
    pe = D.periodicity_data(STAGES_DIAG)
    stages = list(pe["stages"])
    var = pe["variance"]

    def pa(ax):
        ax.plot(stages, var, "o-", color=S.C_GLOBAL, lw=1.8, ms=5.5)
        ax.set_xlabel("Disease stage")
        ax.set_ylabel(r"Var$\,[\langle v \rangle (t)]$"
                      "\n(squared model units)")
        ax.set_xticks(stages)
        ax.set_ylim(0, max(var) * 1.32)
        ax.text(0.97, 0.60,
                f"{var[0]:.2f} $\\rightarrow$ {var[-1]:.2f}\n"
                f"({var[0]/var[-1]:.1f}$\\times$ reduction)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=S.BASE - 2.0, linespacing=1.3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          boxstyle="square,pad=0.2"), zorder=50)
        S.panel_label(ax, "a")

    def pb(ax):
        for i, s in enumerate(stages):
            ax.plot(pe[f"lag_{s}"], pe[f"ac_{s}"],
                    color=S.stage_colour(i, len(stages)), lw=1.5,
                    label=f"Stage {s}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"Lag  $\tau$  (model time units)")
        ax.set_ylabel(r"Normalised autocorrelation  $C(\tau)/C(0)$")
        ax.set_xlim(0, 60)
        ax.set_ylim(-1.0, 1.42)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.legend(loc="upper right", ncol=2, fontsize=S.BASE - 2.5)
        S.panel_label(ax, "b")

    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, 2.55))
    pa(axes[0])
    pb(axes[1])
    fig.tight_layout(w_pad=1.2)
    S.save(fig, "fig09_periodicity_diagnostics")

    return dict(variance=var)


# ==========================================================================
# Figure 10 — percolation transition
# ==========================================================================
def fig10():
    d = np.load(S.PERCOLATION)
    f = d["fractions"]
    Rg_mu, Rg_sd = d["R_global"].mean(0), d["R_global"].std(0)
    Rc_mu, Rc_sd = d["R_component"].mean(0), d["R_component"].std(0)
    LCC = d["LCC"].mean(0)
    LCC_rel = LCC / S.N
    eff = d["efficiency"].mean(0)
    eff_norm = eff / eff[0]
    pr = d["participation_ratio"].mean(0)
    R_pred = Rc_mu * np.sqrt(pr)

    fc_half = D.f_c_by_LCC_half(f, LCC, S.N)
    fc_steep = D.f_c_by_steepest_drop(f, LCC)

    def pa(ax):
        # every quantity here is dimensionless and bounded by 0 and 1, so one
        # axis carries them all and no second scale is needed
        ax.axvspan(fc_steep, fc_half, color="0.55", alpha=0.16, lw=0, zorder=0)
        ax.plot(f, LCC_rel, "-", color=S.C_STRUCT, lw=2.0, zorder=2,
                label=r"LCC$/N$")
        ax.plot(f, eff_norm, ":", color=S.C_EFF, lw=2.0, zorder=2,
                label="efficiency (norm.)")
        ax.errorbar(f, Rg_mu, yerr=Rg_sd, **S.STYLE_GLOBAL, lw=1.8, ms=4.0,
                    capsize=2, zorder=4, label=r"$R_{\rm g}$")
        ax.errorbar(f, Rc_mu, yerr=Rc_sd, **S.STYLE_COMPONENT, lw=1.8, ms=3.8,
                    capsize=2, zorder=4, label=r"$R_{\rm c}$")
        ax.plot(f, R_pred, "-", color=S.C_PRED, lw=1.5, alpha=0.9, zorder=3,
                dashes=(6, 2, 1, 2),
                label=r"$R_{\rm g}^{\rm pred}=R_{\rm c}\sqrt{\rm PR}$")
        ax.annotate(f"percolation window\n$f_c = {fc_steep:.2f}$ to "
                    f"${fc_half:.2f}$",
                    xy=(0.5 * (fc_steep + fc_half), 0.14),
                    xytext=(0.03, 0.04), textcoords="axes fraction",
                    fontsize=S.BASE - 2.5, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"),
                    zorder=40)
        ax.set_xlabel("Edge removal fraction  $f$")
        ax.set_ylabel("Order parameter, LCC$/N$\nand normalised efficiency")
        ax.set_ylim(0, 1.52)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlim(-0.02, 0.92)
        ax.legend(loc="upper right", ncol=2, fontsize=S.BASE - 3.4,
                  columnspacing=0.8, handletextpad=0.45, borderpad=0.3,
                  handlelength=1.7, labelspacing=0.35)
        S.panel_label(ax, "a")

    def pb(ax):
        ax.plot(LCC_rel, Rg_mu, "-", color=S.C_GLOBAL, lw=1.0, alpha=0.35,
                zorder=1)
        ax.plot(LCC_rel, Rc_mu, "--", color=S.C_COMPONENT, lw=1.0, alpha=0.35,
                zorder=1)
        ax.plot(LCC_rel, R_pred, "-", color=S.C_PRED, lw=1.4, alpha=0.85,
                dashes=(6, 2, 1, 2), zorder=2,
                label=r"$R_{\rm g}^{\rm pred}$ (random phases)")
        sc = ax.scatter(LCC_rel, Rg_mu, c=f, cmap="viridis", marker="o", s=42,
                        edgecolors="k", linewidths=0.4, zorder=4,
                        label=r"$R_{\rm g}$")
        ax.scatter(LCC_rel, Rc_mu, c=f, cmap="viridis", marker="s", s=42,
                   edgecolors="k", linewidths=0.4, zorder=3,
                   label=r"$R_{\rm c}$")
        cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Edge removal fraction  $f$")
        cb.ax.tick_params(labelsize=S.BASE - 2.0)
        ax.set_xlabel(r"Relative largest component size  LCC$/N$")
        ax.set_ylabel("Kuramoto order parameter  $R$")
        ax.set_xlim(-0.03, 1.05)
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.legend(loc="lower right", fontsize=S.BASE - 2.2)
        S.panel_label(ax, "b")

    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, 2.75))
    pa(axes[0])
    pb(axes[1])
    fig.tight_layout(w_pad=1.4)
    S.save(fig, "fig10_percolation_diagram")

    return dict(fc_half=fc_half, fc_steep=fc_steep, f=f, Rg=Rg_mu, Rc=Rc_mu,
                R_pred=R_pred, LCC_rel=LCC_rel)


# ==========================================================================
ALL = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05,
       6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10}


# Figures 5 and 7 to 9 read the full (v, w) traces, which are excluded from the
# repository because they run to about 10.7 GB. Regenerate them by setting
# save_trajectories_at in Config and rerunning fhn_alzheimers.py, or skip them.
NEEDS_TRAJECTORIES = {5, 7, 8, 9}


def main(argv):
    wanted = [int(a) for a in argv[1:]] or sorted(ALL)
    have_traj = S.TRAJ_DIR.exists()
    skipped = []
    for n in wanted:
        if n in NEEDS_TRAJECTORIES and not have_traj:
            skipped.append(n)
            continue
        print(f"Figure {n} ...")
        ALL[n]()
    print("done ->", S.FIGDIR)
    if skipped:
        print(f"\nskipped {skipped}: these need the stored (v, w) trajectories,")
        print(f"which are not in the repository. Expected at:\n  {S.TRAJ_DIR}")
        print("Set save_trajectories_at in Config and rerun fhn_alzheimers.py.")


if __name__ == "__main__":
    main(sys.argv)

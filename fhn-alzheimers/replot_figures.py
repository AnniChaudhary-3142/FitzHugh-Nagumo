"""
Rebuilds the four npz-driven figures. Its plotting functions are the ones
build_static_figures.py calls to work straight from runs/.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fhn_alzheimers import Config
from fhn_percolation import (estimate_f_c_by_LCC_half,
                             estimate_f_c_by_steepest_drop)
import fhn_window_convergence as W

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "figure.titlesize": 16, "lines.linewidth": 2.4, "lines.markersize": 8,
    "axes.linewidth": 1.1, "savefig.dpi": 200,
})

RUNS = Path(".")
SCRATCH = Path("replot_preview")


def plot_multi_network(R_global, R_component, cfg, outpath):
    """Faithful copy of fhn_robustness.plot_multi_network, larger markers."""
    n_net, n_stages, n_trials = R_global.shape
    stages = np.arange(1, n_stages + 1)
    R_g_per_net = R_global.mean(axis=2)
    R_c_per_net = R_component.mean(axis=2)
    R_g_mu, R_g_sd = R_g_per_net.mean(axis=0), R_g_per_net.std(axis=0, ddof=1)
    R_c_mu, R_c_sd = R_c_per_net.mean(axis=0), R_c_per_net.std(axis=0, ddof=1)
    R_g_se, R_c_se = R_g_sd / np.sqrt(n_net), R_c_sd / np.sqrt(n_net)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    for i in range(n_net):
        ax.plot(stages, R_g_per_net[i], "-", color="C0", alpha=0.25, lw=1.0)
        ax.plot(stages, R_c_per_net[i], "-", color="C3", alpha=0.25, lw=1.0)
    ax.errorbar(stages, R_g_mu, yerr=R_g_sd, fmt="o-", color="C0",
                lw=2.6, ms=9, capsize=4, label=f"R_global  (mean ± SD, n={n_net})")
    ax.errorbar(stages, R_c_mu, yerr=R_c_sd, fmt="s-", color="C3",
                lw=2.6, ms=9, capsize=4, label="R_component  (mean ± SD)")
    ax.set_xlabel("Disease stage"); ax.set_ylabel("Kuramoto R")
    ax.set_xticks(stages); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

    ax = axes[1]
    ax.errorbar(stages, R_g_mu, yerr=R_g_se, fmt="o-", color="C0",
                lw=2.6, ms=9, capsize=4, label="R_global (mean ± SE)")
    ax.fill_between(stages, R_g_mu - R_g_sd, R_g_mu + R_g_sd,
                    alpha=0.2, color="C0", label="±1 SD across networks")
    ax.errorbar(stages, R_c_mu, yerr=R_c_se, fmt="s-", color="C3",
                lw=2.6, ms=9, capsize=4, label="R_component (mean ± SE)")
    ax.fill_between(stages, R_c_mu - R_c_sd, R_c_mu + R_c_sd,
                    alpha=0.2, color="C3")
    ax.set_xlabel("Disease stage"); ax.set_ylabel("Kuramoto R")
    ax.set_xticks(stages); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_percolation_diagram(data, cfg, out_path):
    """Faithful copy of fhn_percolation.plot_percolation_diagram, larger markers."""
    fractions = data["fractions"]
    R_g_mu, R_g_sd = data["R_global"].mean(0), data["R_global"].std(0)
    R_c_mu, R_c_sd = data["R_component"].mean(0), data["R_component"].std(0)
    LCC_rel = data["LCC"].mean(0) / cfg.N
    eff_norm = data["efficiency"].mean(0) / data["efficiency"].mean(0)[0]
    pr = data["participation_ratio"].mean(0)
    R_pred = R_c_mu * np.sqrt(pr)

    f_c_half = estimate_f_c_by_LCC_half(fractions, data["LCC"].mean(0), cfg.N)
    f_c_steepest = estimate_f_c_by_steepest_drop(fractions, data["LCC"].mean(0))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax = axes[0]; axg = ax.twinx()
    ax.errorbar(fractions, R_g_mu, yerr=R_g_sd, fmt="o-", color="C0",
                lw=2.2, ms=7, capsize=3, label="R_global  (measured)")
    ax.errorbar(fractions, R_c_mu, yerr=R_c_sd, fmt="s-", color="C3",
                lw=2.2, ms=7, capsize=3, label="R_component  (measured)")
    ax.plot(fractions, R_pred, "--", color="C0", alpha=0.55, lw=1.8,
            label=r"$R_{\mathrm{pred}} = R_{\mathrm{comp}}\sqrt{\mathrm{PR}}$"
                  "  (random between-component phases)")
    axg.plot(fractions, LCC_rel, "-", color="C2", lw=2.0, label="LCC / N")
    axg.plot(fractions, eff_norm, ":", color="C4", lw=2.0,
             label="Efficiency (normalised)")
    if f_c_half is not None:
        ax.axvline(f_c_half, color="k", linestyle=":", alpha=0.45, lw=1)
        ax.text(f_c_half + 0.005, 0.04, f"f_c ≈ {f_c_half:.2f}\n(LCC = N/2)",
                fontsize=12)
    ax.axvline(f_c_steepest, color="k", linestyle="--", alpha=0.25, lw=1)
    ax.set_xlabel("Edge removal fraction  f"); ax.set_ylabel("Kuramoto R")
    axg.set_ylabel("LCC / N  and  efficiency (normalised)")
    ax.set_ylim(0, 1.05); axg.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=11)

    ax = axes[1]
    sc = ax.scatter(LCC_rel, R_g_mu, c=fractions, cmap="viridis",
                    s=95, edgecolors="k", linewidths=0.5, zorder=3, label="R_global")
    ax.scatter(LCC_rel, R_c_mu, c=fractions, cmap="viridis", marker="s",
               s=95, edgecolors="k", linewidths=0.5, zorder=2,
               label="R_component", alpha=0.95)
    ax.plot(LCC_rel, R_g_mu, "-", color="C0", lw=1.1, alpha=0.35, zorder=1)
    ax.plot(LCC_rel, R_c_mu, "-", color="C3", lw=1.1, alpha=0.35, zorder=1)
    ax.plot(LCC_rel, R_pred, "--", color="gray", lw=1.5, alpha=0.7, zorder=1,
            label=r"$R_{\mathrm{pred}}$ (random phases)")
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Edge removal fraction  f")
    ax.set_xlabel("LCC / N"); ax.set_ylabel("Kuramoto R")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def do_window(out_path):
    """Copy of fhn_window_convergence.plot_windows at the shared figure sizes."""
    d = np.load(RUNS / "alzheimers_window_convergence_T6400/window_convergence_data.npz")
    dt = float(d["dt"]); block = float(d["block"]); T_long = float(d["T_long"])
    stages = [int(s) for s in d["stages"]]
    traces_g = {s: d[f"R_global_stage{s}"] for s in stages}
    fits = {}
    for s in stages:
        _, bm, _ = W.analyse(traces_g[s], dt, block)
        ft = W.fit_asymptote(bm.mean(axis=0), block, T_long) or {}
        Rc = d[f"R_component_stage{s}"].mean(); PR = d[f"PR_stage{s}"].mean()
        ft["pr_bound"] = float(Rc * np.sqrt(PR)); fits[s] = ft

    TC = W.T_CANONICAL
    fig, axes = plt.subplots(len(stages), 2,
                             figsize=(13.5, 4.6 * len(stages)), squeeze=False)
    for si, stage in enumerate(stages):
        tr = traces_g[stage]; n_rec = tr.shape[2]
        t = np.arange(n_rec) * dt
        prefix, blocks, slopes = W.analyse(tr, dt, block)
        n_blocks = blocks.shape[1]

        ax = axes[si][0]
        flat = tr.reshape(-1, n_rec)
        run = np.cumsum(flat, axis=1) / np.arange(1, n_rec + 1)
        mu, sd = run.mean(axis=0), run.std(axis=0)
        ax.plot(t, mu, color="#1f77b4", lw=2.4, label="running mean")
        ax.fill_between(t, mu - sd, mu + sd, color="#1f77b4", alpha=0.22,
                        label="±1 SD across runs")
        ax.axvline(TC, color="crimson", ls="--", lw=1.8,
                   label=f"canonical window (T={TC:.0f})")
        ax.set_xlabel("Recording time")
        ax.set_ylabel("Running mean of $R_{global}$")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

        ax = axes[si][1]
        tb = (np.arange(n_blocks) + 0.5) * block
        ax.errorbar(tb, blocks.mean(axis=0), yerr=blocks.std(axis=0),
                    fmt="o-", color="#2ca02c", lw=2.4, ms=9, capsize=4,
                    label="block mean")
        ax.errorbar(tb, prefix.mean(axis=0), yerr=prefix.std(axis=0),
                    fmt="s--", color="#7f7f7f", lw=1.6, ms=7, capsize=3,
                    alpha=0.8, label="prefix mean")
        ft = fits.get(stage)
        if ft and "R_inf" in ft:
            tt = np.linspace(0, T_long, 400)
            ax.plot(tt, ft["R_inf"] + ft["A"] * np.exp(-tt / ft["tau"]),
                    color="#d62728", lw=2.0, ls=":",
                    label=(f"fit: $R_\\infty$={ft['R_inf']:.3f}, "
                           f"$\\tau$={ft['tau']:.0f}"))
            ax.axhline(ft["R_inf"], color="#d62728", lw=1.0, alpha=0.5)
        if ft and "pr_bound" in ft:
            ax.axhline(ft["pr_bound"], color="#9467bd", lw=1.6, ls="-.",
                       label=f"PR bound = {ft['pr_bound']:.3f}")
        ax.set_xlabel(f"Recording time (blocks of {block:.0f})")
        ax.set_ylabel("$R_{global}$")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def main(dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)

    # 1. canonical betweenness robustness
    d = np.load(RUNS / "alzheimers_robustness_K0.35_s0.05_a0.65_betweenness"
                        "/robustness_full_data.npz")
    cfg = Config(); cfg.ablation_strategy = "betweenness"
    plot_multi_network(d["R_global"], d["R_component"], cfg,
                       dest / "R_multi_network_robustness.png")

    # 2. random-ablation control
    d = np.load(RUNS / "alzheimers_robustness_random/robustness_results.npz")
    cfg = Config(); cfg.ablation_strategy = "random"
    plot_multi_network(d["R_global"], d["R_component"], cfg,
                       dest / "R_multi_network_random.png")

    # 3. percolation two-panel
    d = np.load(RUNS / "alzheimers_percolation/percolation_data.npz")
    plot_percolation_diagram(d, Config(), dest / "percolation_diagram.png")

    # 4. extended-window convergence
    do_window(dest / "window_convergence.png")

    print("wrote 4 figures to", dest.resolve())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else SCRATCH)

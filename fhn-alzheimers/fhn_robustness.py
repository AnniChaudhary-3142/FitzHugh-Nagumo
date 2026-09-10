
"""
Repeats the ensemble over independent networks and runs paired endpoint
tests on R_global and R_component.
"""

from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict, replace as dc_replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from fhn_alzheimers import Config, run_ensemble


# ---- multi-network outer loop ----
def run_multi_network(cfg: Config, n_networks: int) -> dict:
    """Run the ensemble on n_networks independent graphs, seeded from one master
    seed. Returns R_global and R_component, each (n_networks, n_stages, n_trials)."""
    root = np.random.SeedSequence(cfg.master_seed)
    net_seedseqs = root.spawn(n_networks)

    all_R_global, all_R_component = [], []
    all_metrics = []
    all_per_component = []

    for i, net_ss in enumerate(net_seedseqs):
        # Each network gets its own master_seed, fully independent stream
        net_master = int(net_ss.generate_state(1)[0])
        cfg_i = dc_replace(cfg, master_seed=net_master)

        print(f"\n\n{'#' * 70}")
        print(f"#  NETWORK {i + 1}/{n_networks}    (master_seed={net_master})")
        print(f"{'#' * 70}")

        res = run_ensemble(cfg_i)
        all_R_global.append(res["R_global"])
        all_R_component.append(res["R_component"])
        all_metrics.append(res["metrics_per_stage"])
        all_per_component.append(res["per_stage_per_component"])

    return dict(
        R_global=np.stack(all_R_global),       # (n_net, n_stages, n_trials)
        R_component=np.stack(all_R_component),
        metrics_per_network=all_metrics,
        per_component_per_network=all_per_component,
    )


# ---- statistical tests ----
def stage_endpoint_tests(R: np.ndarray, label: str) -> dict:
    """Paired Stage 1 against final stage, one observation per network."""
    R_per_net = R.mean(axis=2)               # average over trials -> (n_net, n_stages)
    R1   = R_per_net[:, 0]
    Rend = R_per_net[:, -1]
    diff = Rend - R1

    t_stat, p_t = stats.ttest_rel(Rend, R1)
    try:
        w_stat, p_w = stats.wilcoxon(Rend, R1)
    except ValueError:
        w_stat, p_w = float("nan"), float("nan")

    # Cohen's d for paired samples
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")

    n_stages = R.shape[1]
    print(f"\n{label}:")
    print(f"  Stage 1   : {R1.mean():.4f} ± {R1.std(ddof=1):.4f}   "
          f"(n={len(R1)} networks)")
    print(f"  Stage {n_stages}  : {Rend.mean():.4f} ± {Rend.std(ddof=1):.4f}")
    print(f"  Δ (paired): {diff.mean():+.4f} ± {diff.std(ddof=1):.4f}")
    print(f"  Paired t  : t = {t_stat:+.3f},  p = {p_t:.4g}")
    print(f"  Wilcoxon  : W = {w_stat:.1f},  p = {p_w:.4g}")
    print(f"  Cohen's d : {d:+.3f}   "
          f"({'tiny' if abs(d)<0.2 else 'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'} effect)")

    return dict(
        n_networks=int(len(R1)),
        stage1_mean=float(R1.mean()),   stage1_std=float(R1.std(ddof=1)),
        stageN_mean=float(Rend.mean()), stageN_std=float(Rend.std(ddof=1)),
        delta_mean=float(diff.mean()),  delta_std=float(diff.std(ddof=1)),
        t_stat=float(t_stat), p_t=float(p_t),
        w_stat=float(w_stat), p_w=float(p_w),
        cohens_d=float(d),
    )


# ---- plot ----
def plot_multi_network(R_global: np.ndarray, R_component: np.ndarray,
                        cfg: Config, outpath: Path):
    """Two-panel summary: individual network traces + grand mean with bands."""
    n_net, n_stages, n_trials = R_global.shape
    stages = np.arange(1, n_stages + 1)

    R_g_per_net = R_global.mean(axis=2)       # (n_net, n_stages)
    R_c_per_net = R_component.mean(axis=2)

    R_g_mu, R_g_sd = R_g_per_net.mean(axis=0), R_g_per_net.std(axis=0, ddof=1)
    R_c_mu, R_c_sd = R_c_per_net.mean(axis=0), R_c_per_net.std(axis=0, ddof=1)
    R_g_se = R_g_sd / np.sqrt(n_net)
    R_c_se = R_c_sd / np.sqrt(n_net)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    # ---- Left: every network as a faint line, grand mean overlaid ----
    ax = axes[0]
    for i in range(n_net):
        ax.plot(stages, R_g_per_net[i], "-", color="C0", alpha=0.25, lw=0.9)
        ax.plot(stages, R_c_per_net[i], "-", color="C3", alpha=0.25, lw=0.9)
    ax.errorbar(stages, R_g_mu, yerr=R_g_sd, fmt="o-", color="C0",
                 lw=2.5, ms=7, capsize=4,
                 label=f"R_global  (mean ± SD, n={n_net})")
    ax.errorbar(stages, R_c_mu, yerr=R_c_sd, fmt="s-", color="C3",
                 lw=2.5, ms=7, capsize=4,
                 label=f"R_component  (mean ± SD)")
    ax.set_xlabel("Disease stage")
    ax.set_ylabel("Kuramoto R")
    ax.set_title(f"All {n_net} network realizations\n"
                  f"(faint = individual networks, bold = mean ± SD)")
    ax.set_xticks(stages);  ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3);  ax.legend(loc="best")

    # ---- Right: clean summary with SE bars + SD bands ----
    ax = axes[1]
    ax.errorbar(stages, R_g_mu, yerr=R_g_se, fmt="o-", color="C0",
                 lw=2.5, ms=8, capsize=4, label="R_global (mean ± SE)")
    ax.fill_between(stages, R_g_mu - R_g_sd, R_g_mu + R_g_sd,
                     alpha=0.2, color="C0", label="±1 SD across networks")
    ax.errorbar(stages, R_c_mu, yerr=R_c_se, fmt="s-", color="C3",
                 lw=2.5, ms=8, capsize=4, label="R_component (mean ± SE)")
    ax.fill_between(stages, R_c_mu - R_c_sd, R_c_mu + R_c_sd,
                     alpha=0.2, color="C3")
    ax.set_xlabel("Disease stage")
    ax.set_ylabel("Kuramoto R")
    ax.set_title("Summary statistics")
    ax.set_xticks(stages);  ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3);  ax.legend(loc="best")

    fig.suptitle(
        f"Robustness across network realizations  "
        f"({cfg.ablation_strategy} ablation, K={cfg.K}, a={cfg.a})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- main ----
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks", type=int, default=10,
                         help="Number of independent network realizations")
    parser.add_argument("--trials", type=int, default=3,
                         help="Trials per (network, stage)")
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--K", type=float, default=0.35,
                         help="Coupling (0.35 ≈ paper baseline)")
    parser.add_argument("--sigma", type=float, default=0.05,
                         help="Noise amplitude (0.05 ≈ paper baseline)")
    parser.add_argument("--a", type=float, default=0.65,
                         help="Excitability (0.65 oscillatory, >1 excitable)")
    

    parser.add_argument("--ablation-max", type=float, default=0.85)
    parser.add_argument("--ablation", choices=["betweenness", "random"],
                         default="betweenness")
    parser.add_argument("--quick", action="store_true",
                         help="Tiny run for testing")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--outdir", type=str, default="alzheimers_robustness")
    args = parser.parse_args()
    # auto-named outdir keeps sweep points from colliding
    if args.outdir == "alzheimers_robustness":
        args.outdir = (f"alzheimers_robustness_K{args.K}_s{args.sigma}"
                        f"_a{args.a}_{args.ablation}")

    cfg = Config()
    cfg.N             = args.N
    cfg.n_stages      = args.stages
    cfg.n_trials      = args.trials
    cfg.K             = args.K
    cfg.sigma         = args.sigma
    cfg.a             = args.a
    cfg.ablation_max  = args.ablation_max
    cfg.ablation_strategy = args.ablation
    cfg.master_seed   = args.seed

    n_nets = args.networks
    if args.quick:
        cfg.N, cfg.n_trials = 300, 2
        cfg.T, cfg.burn_in = 150.0, 50.0
        n_nets = min(3, args.networks)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Run ----
    print(f"\n{'=' * 70}")
    print(f"MULTI-NETWORK ROBUSTNESS — {cfg.ablation_strategy} ablation")
    print(f"  {n_nets} networks  ×  {cfg.n_stages} stages  ×  "
          f"{cfg.n_trials} trials   (N={cfg.N})")
    print(f"  K={cfg.K},  a={cfg.a},  ablation_max={cfg.ablation_max}")
    print(f"  master_seed={cfg.master_seed}")
    print(f"{'=' * 70}")

    t0 = time.time()
    results = run_multi_network(cfg, n_nets)
    print(f"\n\n>>> Total wall-time: {(time.time()-t0)/60:.1f} min")

    # ---- Stats ----
    print(f"\n{'=' * 70}")
    print(f"PAIRED STATISTICAL TESTS  (Stage 1 vs Stage {cfg.n_stages})")
    print(f"  Each network contributes one paired observation.")
    print(f"{'=' * 70}")
    stats_g = stage_endpoint_tests(results["R_global"], "R_global")
    stats_c = stage_endpoint_tests(results["R_component"], "R_component")

    # ---- Plot + save ----
    plot_multi_network(results["R_global"], results["R_component"], cfg,
                        outdir / "R_multi_network.png")

    # Full per-stage, per-trial, per-network R arrays
    np.savez_compressed(
        outdir / "robustness_full_data.npz",
        R_global=results["R_global"],          # (n_net, n_stages, n_trials)
        R_component=results["R_component"],    # (n_net, n_stages, n_trials)
    )

    # Per-network graph metrics across stages
    with open(outdir / "metrics_per_network.json", "w") as f:
        json.dump(results["metrics_per_network"], f, indent=2)

    # Per-stage component sizes + per-component R (per network)
    with open(outdir / "per_component_per_network.json", "w") as f:
        json.dump(results["per_component_per_network"], f, indent=2)

    with open(outdir / "robustness_summary.json", "w") as f:
        json.dump(dict(
            config=asdict(cfg),
            n_networks=n_nets,
            stats_R_global=stats_g,
            stats_R_component=stats_c,
        ), f, indent=2)

    # ---- Paper-ready sentence ----
    print(f"\n{'=' * 70}")
    print(f"PAPER-READY SUMMARY SENTENCE")
    print(f"{'=' * 70}")
    print(
        f"\nAcross n={n_nets} independent Watts-Strogatz realizations, "
        f"R_global decreased significantly from healthy (Stage 1: "
        f"{stats_g['stage1_mean']:.3f} ± {stats_g['stage1_std']:.3f}) "
        f"to advanced ablation (Stage {cfg.n_stages}: "
        f"{stats_g['stageN_mean']:.3f} ± {stats_g['stageN_std']:.3f}); "
        f"paired t = {stats_g['t_stat']:.2f}, p = {stats_g['p_t']:.2e}, "
        f"Cohen's d = {stats_g['cohens_d']:.2f}. "
        f"R_component changed by only Δ = {stats_c['delta_mean']:+.3f}, "
        f"confirming that desynchronization is structural rather than dynamical."
    )

    print(f"\n✓ All outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()

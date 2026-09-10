"""
Fine-resolution edge-removal sweep at canonical parameters, resolving the
percolation-like transition and the decorrelated-phase reference level.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from fhn_alzheimers import (Config, build_initial_network, simulate,
                              compute_graph_metrics)


# ---- sweep ----
def run_percolation_sweep(cfg: Config, fractions, n_networks: int) -> dict:
    n_f = len(fractions)
    arrays = dict(
        R_global=np.zeros((n_networks, n_f)),
        R_component=np.zeros((n_networks, n_f)),
        LCC=np.zeros((n_networks, n_f), dtype=np.int64),
        n_components=np.zeros((n_networks, n_f), dtype=np.int64),
        efficiency=np.zeros((n_networks, n_f)),
        clustering=np.zeros((n_networks, n_f)),
        modularity=np.zeros((n_networks, n_f)),
        participation_ratio=np.zeros((n_networks, n_f)),
    )

    root = np.random.SeedSequence(cfg.master_seed)
    net_seedseqs = root.spawn(n_networks)

    for i, net_ss in enumerate(net_seedseqs):
        net_seed = int(net_ss.generate_state(1)[0])
        print(f"\n=== NETWORK {i+1}/{n_networks}  (seed = {net_seed}) ===")
        G0 = build_initial_network(cfg, net_seed)

        print(f"  Computing edge-betweenness on healthy graph ...")
        t0 = time.time()
        eb = nx.edge_betweenness_centrality(G0)
        edge_order = sorted(eb, key=eb.get, reverse=True)
        print(f"    done in {time.time()-t0:.1f}s")

        for j, frac in enumerate(fractions):
            n_remove = int(round(frac * len(edge_order)))
            G = G0.copy()
            G.remove_edges_from(edge_order[:n_remove])

            m = compute_graph_metrics(G)
            sizes = [len(c) for c in nx.connected_components(G)]
            pr = sum(s * s for s in sizes) / (cfg.N * cfg.N)

            arrays["LCC"][i, j]              = m["largest_cc_size"]
            arrays["n_components"][i, j]     = m["n_components"]
            arrays["efficiency"][i, j]       = m["global_efficiency"]
            arrays["clustering"][i, j]       = m["mean_clustering"]
            arrays["modularity"][i, j]       = m["modularity"]
            arrays["participation_ratio"][i, j] = pr

            trial_seed = net_seed * 1000 + j      # unique seed per (network, f)
            t0 = time.time()
            res = simulate(G, cfg, seed=trial_seed, store_full=False)
            arrays["R_global"][i, j]    = res["R_global_mean"]
            arrays["R_component"][i, j] = res["R_component_mean"]

            print(f"  f = {frac:.3f}   |E|={m['n_edges']:>4}   "
                  f"LCC={m['largest_cc_size']:>4}   PR={pr:.4f}   "
                  f"R_g={arrays['R_global'][i,j]:.3f}   "
                  f"R_c={arrays['R_component'][i,j]:.3f}   "
                  f"({time.time()-t0:.1f}s)")

    arrays["fractions"] = fractions
    return arrays


# ---- percolation threshold estimation ----
def estimate_f_c_by_LCC_half(fractions, LCC_mean, N) -> float | None:
    """Fraction at which the mean largest component first drops below N/2,
    interpolated between bracketing points."""
    rel = LCC_mean / N
    above = np.where(rel >= 0.5)[0]
    below = np.where(rel < 0.5)[0]
    if len(above) == 0 or len(below) == 0:
        return None
    i1, i2 = above[-1], below[0]
    if i2 - i1 != 1:
        return None
    f1, f2 = fractions[i1], fractions[i2]
    L1, L2 = rel[i1], rel[i2]
    return float(f1 + (f2 - f1) * (0.5 - L1) / (L2 - L1))


def estimate_f_c_by_steepest_drop(fractions, LCC_mean) -> float:
    """f_c defined as the fraction where dLCC/df is most negative."""
    deriv = np.gradient(LCC_mean, fractions)
    idx = int(np.argmin(deriv))
    return float(fractions[idx])


# ---- plot ----
def plot_percolation_diagram(data, cfg: Config, out_path):
    fractions = data["fractions"]
    R_g_mu, R_g_sd = data["R_global"].mean(0), data["R_global"].std(0)
    R_c_mu, R_c_sd = data["R_component"].mean(0), data["R_component"].std(0)
    LCC_rel = data["LCC"].mean(0) / cfg.N
    eff_norm = data["efficiency"].mean(0) / data["efficiency"].mean(0)[0]
    pr = data["participation_ratio"].mean(0)

    R_pred = R_c_mu * np.sqrt(pr)            # fully-decorrelated lower bound

    f_c_half     = estimate_f_c_by_LCC_half(fractions, data["LCC"].mean(0), cfg.N)
    f_c_steepest = estimate_f_c_by_steepest_drop(fractions, data["LCC"].mean(0))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # ---- Panel A: bifurcation-style view ----
    ax  = axes[0]
    axg = ax.twinx()

    ax.errorbar(fractions, R_g_mu, yerr=R_g_sd, fmt="o-", color="C0",
                lw=2, ms=4.5, capsize=2.5, label="R_global  (measured)")
    ax.errorbar(fractions, R_c_mu, yerr=R_c_sd, fmt="s-", color="C3",
                lw=2, ms=4.5, capsize=2.5, label="R_component  (measured)")
    ax.plot(fractions, R_pred, "--", color="C0", alpha=0.55, lw=1.5,
             label=r"$R_{\mathrm{pred}} = R_{\mathrm{comp}}\sqrt{\mathrm{PR}}$"
                   "  (random between-component phases)")

    axg.plot(fractions, LCC_rel, "-",  color="C2", lw=1.7, label="LCC / N")
    axg.plot(fractions, eff_norm, ":", color="C4", lw=1.7,
              label="Efficiency (normalised)")

    if f_c_half is not None:
        ax.axvline(f_c_half, color="k", linestyle=":", alpha=0.45, lw=1)
        ax.text(f_c_half + 0.005, 0.04,
                f"f_c ≈ {f_c_half:.2f}\n(LCC = N/2)",
                fontsize=9)
    ax.axvline(f_c_steepest, color="k", linestyle="--", alpha=0.25, lw=1)

    ax.set_xlabel("Edge removal fraction  f")
    ax.set_ylabel("Kuramoto R")
    axg.set_ylabel("LCC / N  and  efficiency (normalised)")
    ax.set_title(f"Bifurcation view: dynamics and structure vs f\n"
                  f"(K = {cfg.K}, σ = {cfg.sigma}, a = {cfg.a},  N = {cfg.N})")
    ax.set_ylim(0, 1.05)
    axg.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.5)

    # ---- Panel B: structural-dynamical correspondence (R vs LCC) ----
    ax = axes[1]
    sc = ax.scatter(LCC_rel, R_g_mu, c=fractions, cmap="viridis",
                     s=55, edgecolors="k", linewidths=0.4, zorder=3,
                     label="R_global")
    ax.scatter(LCC_rel, R_c_mu, c=fractions, cmap="viridis",
                marker="s", s=55, edgecolors="k", linewidths=0.4,
                zorder=2, label="R_component", alpha=0.95)
    ax.plot(LCC_rel, R_g_mu, "-", color="C0", lw=0.9, alpha=0.35, zorder=1)
    ax.plot(LCC_rel, R_c_mu, "-", color="C3", lw=0.9, alpha=0.35, zorder=1)
    ax.plot(LCC_rel, R_pred, "--", color="gray", lw=1.2, alpha=0.7,
             zorder=1, label=r"$R_{\mathrm{pred}}$ (random phases)")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Edge removal fraction  f")
    ax.set_xlabel("LCC / N")
    ax.set_ylabel("Kuramoto R")
    ax.set_title("Structural–dynamical correspondence\n"
                  "(parametric in f, points coloured by ablation)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return f_c_half, f_c_steepest


# ---- main ----
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fracs",  type=int, default=30)
    parser.add_argument("--networks", type=int, default=2)
    parser.add_argument("--f-min",    type=float, default=0.0)
    parser.add_argument("--f-max",    type=float, default=0.90)
    parser.add_argument("--quick",    action="store_true")
    parser.add_argument("--outdir",   type=str, default="alzheimers_percolation")
    parser.add_argument("--seed",     type=int, default=20260529)
    args = parser.parse_args()

    cfg = Config()
    cfg.master_seed = args.seed
    n_fracs    = args.n_fracs
    n_networks = args.networks
    if args.quick:
        n_fracs, n_networks = 15, 1
        cfg.T, cfg.burn_in = 200.0, 50.0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fractions = np.linspace(args.f_min, args.f_max, n_fracs)

    print(f"\nPERCOLATION SWEEP  —  canonical parameters")
    print(f"  K = {cfg.K},  σ = {cfg.sigma},  a = {cfg.a},  N = {cfg.N}")
    print(f"  {n_fracs} f-values from {args.f_min} to {args.f_max}")
    print(f"  {n_networks} network realisation(s)")
    print(f"  master_seed = {cfg.master_seed}")

    t_start = time.time()
    data = run_percolation_sweep(cfg, fractions, n_networks)
    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\nTotal wall-time: {elapsed_min:.1f} min")

    np.savez_compressed(outdir / "percolation_data.npz", **data)

    f_c_half, f_c_steepest = plot_percolation_diagram(
        data, cfg, outdir / "percolation_diagram.png"
    )

    summary = dict(
        config={"K": cfg.K, "sigma": cfg.sigma, "a": cfg.a, "N": cfg.N,
                "master_seed": cfg.master_seed,
                "n_fractions": int(n_fracs), "n_networks": int(n_networks)},
        f_c_LCC_half=f_c_half,
        f_c_steepest_drop=f_c_steepest,
        f_min=float(args.f_min), f_max=float(args.f_max),
        R_global_at_f0=float(data["R_global"].mean(0)[0]),
        R_global_at_fmax=float(data["R_global"].mean(0)[-1]),
        R_component_at_f0=float(data["R_component"].mean(0)[0]),
        R_component_at_fmax=float(data["R_component"].mean(0)[-1]),
        LCC_at_f0=float(data["LCC"].mean(0)[0]),
        LCC_at_fmax=float(data["LCC"].mean(0)[-1]),
    )
    with open(outdir / "percolation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n>>> f_c (LCC crosses N/2)   ≈ {f_c_half:.3f}"
          if f_c_half is not None else "\n>>> f_c (LCC crosses N/2): not bracketed")
    print(f">>> f_c (steepest dLCC/df)  ≈ {f_c_steepest:.3f}")
    print(f"\n✓ All outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()

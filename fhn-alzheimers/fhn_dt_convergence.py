"""
Reruns one network at dt = 0.02, 0.01 and 0.005 to see how far R moves
when the timestep is halved and doubled.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from fhn_alzheimers import Config, build_initial_network, simulate


# ---- config ----
DT_VALUES   = (0.02, 0.01, 0.005)
N_TRIALS    = 3
OUT_DIR     = Path("alzheimers_dt_convergence")
CANONICAL_DT = 0.01


# ---- sweep ----
def run_convergence_sweep():
    cfg_base = Config()

    # one network and one betweenness ordering, shared across all dt
    root = np.random.SeedSequence(cfg_base.master_seed)
    net_ss = root.spawn(1)[0]
    net_seed = int(net_ss.generate_state(1)[0])

    G0 = build_initial_network(cfg_base, net_seed)
    print(f"Computing edge-betweenness on healthy graph (one-shot)...")
    t0 = time.time()
    eb = nx.edge_betweenness_centrality(G0)
    edge_order = sorted(eb, key=eb.get, reverse=True)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  |E|_0 = {len(edge_order)}")

    n_stages = cfg_base.n_stages
    n_dt = len(DT_VALUES)

    R_global    = np.zeros((n_dt, n_stages, N_TRIALS))
    R_component = np.zeros((n_dt, n_stages, N_TRIALS))

    for di, dt in enumerate(DT_VALUES):
        cfg = Config()
        cfg.dt = dt
        cfg.master_seed = cfg_base.master_seed
        cfg.raster_stages = ()           # no raster overhead
        cfg.save_trajectories_at = ()    # no trajectory I/O
        cfg.trajectory_outdir = ""

        print(f"\n=== Δt = {dt} {'(canonical)' if abs(dt-CANONICAL_DT)<1e-12 else ''} ===")

        for s in range(n_stages):
            alpha = s / (n_stages - 1) if n_stages > 1 else 0.0
            frac = alpha * cfg.ablation_max
            n_remove = int(round(frac * len(edge_order)))

            G = G0.copy()
            G.remove_edges_from(edge_order[:n_remove])

            stage_times = []
            for k in range(N_TRIALS):
                trial_seed = net_seed * 10000 + di * 1000 + s * 100 + k
                t0 = time.time()
                res = simulate(G, cfg, seed=trial_seed, store_full=False)
                R_global[di, s, k]    = res["R_global_mean"]
                R_component[di, s, k] = res["R_component_mean"]
                stage_times.append(time.time() - t0)

            mu_g = R_global[di, s].mean()
            mu_c = R_component[di, s].mean()
            print(f"  Stage {s+1:>2}/{n_stages}: |E|={G.number_of_edges():>4}  "
                  f"R_g={mu_g:.4f}  R_c={mu_c:.4f}  "
                  f"(avg {np.mean(stage_times):.1f}s/trial)")

    return R_global, R_component, net_seed


# ---- plot ----
def plot_convergence(R_global, R_component, out_path):
    n_dt, n_stages, _ = R_global.shape
    stages = np.arange(1, n_stages + 1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_dt))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for di, dt in enumerate(DT_VALUES):
        mu_g, sd_g = R_global[di].mean(axis=1),    R_global[di].std(axis=1)
        mu_c, sd_c = R_component[di].mean(axis=1), R_component[di].std(axis=1)
        is_canon = abs(dt - CANONICAL_DT) < 1e-12
        lw = 2.4 if is_canon else 1.7
        ls = "-" if is_canon else "--"
        label = f"Δt = {dt}" + ("  (canonical)" if is_canon else "")

        axes[0].errorbar(stages, mu_g, yerr=sd_g, fmt=f"o{ls}",
                         color=colors[di], lw=lw, ms=5.5, capsize=3,
                         label=label)
        axes[1].errorbar(stages, mu_c, yerr=sd_c, fmt=f"s{ls}",
                         color=colors[di], lw=lw, ms=5.5, capsize=3,
                         label=label)

    for ax, ttl in zip(axes, ["R_global", "R_component"]):
        ax.set_xlabel("Disease stage")
        ax.set_ylabel(ttl)
        ax.set_xticks(stages)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=10)

    axes[0].set_title("Global Kuramoto order parameter")
    axes[1].set_title("Component-restricted Kuramoto order parameter")
    fig.suptitle(
        "Timestep-convergence check: R at Δt = 0.02, 0.01, 0.005",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- main ----
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nΔt CONVERGENCE CHECK")
    print(f"  Δt values: {DT_VALUES}")
    print(f"  N_TRIALS per (Δt, stage): {N_TRIALS}")

    t_start = time.time()
    R_global, R_component, net_seed = run_convergence_sweep()
    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\nTotal wall-time: {elapsed_min:.1f} min")

    np.savez_compressed(
        OUT_DIR / "dt_convergence_data.npz",
        dt_values=np.array(DT_VALUES),
        R_global=R_global,
        R_component=R_component,
        net_seed=net_seed,
    )

    plot_convergence(R_global, R_component, OUT_DIR / "dt_convergence.png")

    # ----- summary -----
    canonical_idx = DT_VALUES.index(CANONICAL_DT)
    R_g_canon = R_global[canonical_idx].mean(axis=1)
    R_c_canon = R_component[canonical_idx].mean(axis=1)

    summary = dict(
        dt_values=list(DT_VALUES),
        canonical_dt=CANONICAL_DT,
        n_trials=N_TRIALS,
        net_seed=int(net_seed),
        deviations={},
    )

    print("\n--- Stage-by-stage R_global (mean across trials) ---")
    header = "Stage | " + " | ".join(f"Δt={dt}" for dt in DT_VALUES)
    print(header)
    print("-" * len(header))
    for s in range(R_global.shape[1]):
        row = f"  {s+1:>2}  | " + " | ".join(
            f"{R_global[di, s].mean():.4f}" for di in range(len(DT_VALUES))
        )
        print(row)

    print("\n--- Stage-by-stage R_component (mean across trials) ---")
    print(header)
    print("-" * len(header))
    for s in range(R_global.shape[1]):
        row = f"  {s+1:>2}  | " + " | ".join(
            f"{R_component[di, s].mean():.4f}" for di in range(len(DT_VALUES))
        )
        print(row)

    print("\n--- Max |Δ R| vs canonical Δt=0.01 ---")
    for di, dt in enumerate(DT_VALUES):
        if di == canonical_idx:
            continue
        diff_g = R_global[di].mean(axis=1) - R_g_canon
        diff_c = R_component[di].mean(axis=1) - R_c_canon
        max_g = float(np.max(np.abs(diff_g)))
        max_c = float(np.max(np.abs(diff_c)))
        st_g = int(np.argmax(np.abs(diff_g))) + 1
        st_c = int(np.argmax(np.abs(diff_c))) + 1
        print(f"  Δt = {dt:>5}: max |ΔR_global|    = {max_g:.4f}  @ stage {st_g}")
        print(f"            max |ΔR_component| = {max_c:.4f}  @ stage {st_c}")
        summary["deviations"][str(dt)] = dict(
            max_abs_R_global=max_g,
            max_abs_R_component=max_c,
            stage_max_R_global=st_g,
            stage_max_R_component=st_c,
        )

    with open(OUT_DIR / "dt_convergence_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n✓ All outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

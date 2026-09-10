
"""
Stochastic FitzHugh-Nagumo network model of Alzheimer's progression.
Neurons stay fixed in the oscillatory regime while high-betweenness edges
are ablated stage by stage. Every other script imports this one.
"""

from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy import sparse


# ---- configuration ----
@dataclass
class Config:
    N: int = 1000
    k_nbr: int = 8
    p_rewire: float = 0.1

    # fixed across stages: neurons oscillate alone, so synchrony is from coupling
    a: float = 0.65
    eps: float = 0.06
    I_ext: float = 0.30
    b: float = 0.80
    K: float = 0.35            # coupling strength, fixed across stages

    # 0 is homogeneous; above 0 each neuron draws a ~ N(a, a_spread*a)
    a_spread: float = 0.0
    a_seed: int = 0

    # detunes frequency without moving the distance to the bifurcation
    eps_spread: float = 0.0
    eps_seed: int = 0

    T: float = 400.0
    burn_in: float = 100.0
    dt: float = 0.01
    K_ramp_time: float = 20.0
    sigma: float = 0.05

    # edge ablation schedule
    n_stages: int = 10
    ablation_max: float = 0.85         # Stage n removes this fraction of edges
    ablation_strategy: str = "betweenness"   # "betweenness" or "random"
    recompute_betweenness: bool = False      # True = expensive but biologically richer

    n_trials: int = 5
    raster_stages: tuple = (1, 5, 10)

    save_trajectories_at: tuple = (1,3,5,7,10)           # stage numbers; empty = don't persist
    trajectory_outdir: str = "alzheimers_results/trajectories_canonical"              # directory for .npz trajectory files

    master_seed: int = 20260529

    def ablation_fraction(self, stage: int) -> float:
        if self.n_stages == 1:
            return 0.0
        alpha = (stage - 1) / (self.n_stages - 1)
        return alpha * self.ablation_max


# ---- FHN core ----
def fhn_fixed_point(a: float, b: float, I: float) -> tuple[float, float]:
    def f(v): return v - v**3 / 3.0 + I - (v + a) / b
    grid = np.linspace(-3.0, 3.0, 601)
    sign_changes = np.where(np.diff(np.sign(f(grid))))[0]
    if len(sign_changes) == 0:
        raise ValueError(f"No fixed point for a={a}, b={b}, I={I}")
    idx = sign_changes[0]
    v_star = brentq(f, grid[idx], grid[idx + 1])
    w_star = (v_star + a) / b
    return v_star, w_star


def fhn_drift(v, w, A_csr, deg, K, I, a, b, eps):
    deg_safe = np.where(deg > 0, deg, 1.0)   # avoid div-by-0 for isolated nodes (sum term is 0 there anyway)
    coupling = K * (A_csr @ v - deg * v) / deg_safe
    dv = v - v**3 / 3.0 - w + I + coupling
    dw = eps * (v + a - b * w)
    return dv, dw


# ---- network construction and ablation ----
def build_initial_network(cfg: Config, seed: int) -> nx.Graph:
    """Healthy (Stage 1) Watts-Strogatz small-world graph."""
    return nx.watts_strogatz_graph(cfg.N, k=cfg.k_nbr, p=cfg.p_rewire, seed=seed)


def ablate_by_betweenness(G: nx.Graph, fraction: float,
                          edge_order: list | None = None) -> tuple[nx.Graph, list]:
    """Remove the highest-betweenness `fraction` of edges. Reuses edge_order when
    given. Returns (graph, edge_order)."""
    if edge_order is None:
        eb = nx.edge_betweenness_centrality(G)
        edge_order = sorted(eb, key=eb.get, reverse=True)
    n_remove = int(round(fraction * len(edge_order)))
    G_abl = G.copy()
    G_abl.remove_edges_from(edge_order[:n_remove])
    return G_abl, edge_order


def ablate_random(G: nx.Graph, fraction: float, rng: np.random.Generator
                  ) -> nx.Graph:
    """Control: remove `fraction` of edges uniformly at random."""
    edges = list(G.edges())
    n_remove = int(round(fraction * len(edges)))
    perm = rng.permutation(len(edges))
    G_abl = G.copy()
    G_abl.remove_edges_from(edges[i] for i in perm[:n_remove])
    return G_abl


def graph_to_sparse(G: nx.Graph) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Convert nx.Graph to (A_csr, deg) for the integrator."""
    A_csr = sparse.csr_matrix(nx.to_scipy_sparse_array(G, dtype=np.float64,
                                                         nodelist=sorted(G.nodes())))
    deg = np.asarray(A_csr.sum(axis=1)).flatten()
    return A_csr, deg


# ---- graph metrics ----
def compute_graph_metrics(G: nx.Graph) -> dict:
    """All the graph metrics for the disease progression panels."""
    components = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in components), reverse=True)
    largest = max(components, key=len)
    G_largest = G.subgraph(largest)

    metrics = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": len(components),
        "largest_cc_size": len(largest),
        "fragmentation": 1.0 - len(largest) / G.number_of_nodes(),
        "mean_clustering": nx.average_clustering(G),
        "global_efficiency": nx.global_efficiency(G),
    }
    # Path length only meaningful on the largest component
    if len(largest) > 1:
        metrics["avg_path_length_lcc"] = nx.average_shortest_path_length(G_largest)
    else:
        metrics["avg_path_length_lcc"] = 0.0

    try:
        communities = nx.community.greedy_modularity_communities(G)
        metrics["modularity"] = nx.community.modularity(G, communities)
        metrics["n_communities"] = len(communities)
    except Exception:
        metrics["modularity"] = float("nan")
        metrics["n_communities"] = -1
    return metrics


# ---- simulation, with R_global and R_component computed online ----
def simulate(G: nx.Graph, cfg: Config, seed: int,
             store_full: bool = False) -> dict:
    """One trial. Returns time-averaged R_global and R_component, and the full
    (V, W) trajectories when asked."""
    rng = np.random.default_rng(seed)

    nodes_sorted = sorted(G.nodes())
    N = len(nodes_sorted)
    A_csr, deg = graph_to_sparse(G)

    # Components in the sparse-matrix index space
    node_to_idx = {n: i for i, n in enumerate(nodes_sorted)}
    components = [np.array([node_to_idx[n] for n in c], dtype=np.int64)
                   for c in nx.connected_components(G)]
    big_components = [c for c in components if len(c) >= 2]
    big_sizes = np.array([len(c) for c in big_components])
    big_total = big_sizes.sum() if len(big_sizes) else 0

    # Per-neuron excitability. Scalar reproduces the homogeneous model.
    if cfg.a_spread > 0.0:
        a_use = np.random.default_rng(cfg.a_seed).normal(
            cfg.a, cfg.a_spread * cfg.a, size=N)
    else:
        a_use = cfg.a

    # Per-neuron timescale, i.e. natural-frequency detuning.
    if cfg.eps_spread > 0.0:
        eps_use = np.random.default_rng(cfg.eps_seed).normal(
            cfg.eps, cfg.eps_spread * cfg.eps, size=N)
        if np.any(eps_use <= 0.0):
            raise ValueError("eps_spread produced a non-positive eps")
    else:
        eps_use = cfg.eps

    # each neuron orbits its own fixed point, so the phase reference is per-neuron
    if np.isscalar(a_use):
        v_star, w_star = fhn_fixed_point(a_use, cfg.b, cfg.I_ext)
    else:
        _fp = [fhn_fixed_point(ai, cfg.b, cfg.I_ext) for ai in a_use]
        v_star = np.array([p[0] for p in _fp])
        w_star = np.array([p[1] for p in _fp])
    v = v_star + 0.05 * rng.standard_normal(N)
    w = w_star + 0.05 * rng.standard_normal(N)

    n_total  = int(round((cfg.T + cfg.burn_in) / cfg.dt))
    n_burn   = int(round(cfg.burn_in / cfg.dt))
    n_record = n_total - n_burn
    n_ramp   = max(int(round(cfg.K_ramp_time / cfg.dt)), 1)
    sqrt_dt  = np.sqrt(cfg.dt)

    R_global_trace = np.empty(n_record)
    R_component_trace = np.empty(n_record)

    # Accumulator for per-component R (time-averaged at end)
    n_big = len(big_components)
    R_per_comp_accum = np.zeros(n_big) if n_big > 0 else np.array([])

    if store_full:
        V = np.empty((n_record, N), dtype=np.float32)
        W = np.empty((n_record, N), dtype=np.float32)

    K_target = cfg.K
    I = cfg.I_ext
    for t in range(n_total):
        K = K_target * min(1.0, t / n_ramp)
        dW = sqrt_dt * rng.standard_normal(N)

        dv1, dw1 = fhn_drift(v, w, A_csr, deg, K, I, a_use, cfg.b, eps_use)
        v_p = v + cfg.dt * dv1 + cfg.sigma * dW
        w_p = w + cfg.dt * dw1
        dv2, dw2 = fhn_drift(v_p, w_p, A_csr, deg, K, I, a_use, cfg.b, eps_use)
        v = v + 0.5 * cfg.dt * (dv1 + dv2) + cfg.sigma * dW
        w = w + 0.5 * cfg.dt * (dw1 + dw2)

        if t % 2000 == 0 and not np.all(np.isfinite(v)):
            raise RuntimeError(f"Integrator unstable at step {t}")

        if t >= n_burn:
            i = t - n_burn
            phase = np.arctan2(w - w_star, v - v_star)
            # Global R: across all N neurons regardless of connectivity
            R_global_trace[i] = np.abs(np.mean(np.exp(1j * phase)))
            # Component R: weighted average over components of size >= 2
            if big_total > 0:
                R_w = 0.0
                for j, (cs, idx) in enumerate(zip(big_sizes, big_components)):
                    z = np.mean(np.exp(1j * phase[idx]))
                    R_k = np.abs(z)
                    R_w += R_k * cs
                    R_per_comp_accum[j] += R_k
                R_component_trace[i] = R_w / big_total
            else:
                R_component_trace[i] = 0.0
            if store_full:
                V[i] = v
                W[i] = w

    out = dict(
        R_global_mean=float(R_global_trace.mean()),
        R_global_std=float(R_global_trace.std()),
        R_component_mean=float(R_component_trace.mean()),
        R_component_std=float(R_component_trace.std()),
        R_global_trace=R_global_trace,
        R_component_trace=R_component_trace,
        fixed_point=(v_star, w_star),
        R_per_component=(R_per_comp_accum / n_record) if n_big > 0 else np.array([]),
        component_sizes=big_sizes.copy() if n_big > 0 else np.array([], dtype=np.int64),
        n_isolates=int(N - big_total),
    )
    if store_full:
        out["V"] = V
        out["W"] = W
    return out


# ---- ensemble experiment ----
def run_ensemble(cfg: Config):
    print(f"\n{'='*64}")
    print(f"Alzheimer's FHN network — disease progression by ablation")
    print(f"{'='*64}")
    print(f"  N={cfg.N}  K={cfg.K}  a={cfg.a}  ε={cfg.eps}  I={cfg.I_ext}")
    print(f"  stages={cfg.n_stages}  trials={cfg.n_trials}  "
          f"ablation_max={cfg.ablation_max}  strategy={cfg.ablation_strategy}")
    print(f"  master_seed={cfg.master_seed}")

    root = np.random.SeedSequence(cfg.master_seed)
    net_ss, abl_ss, *trial_ss = root.spawn(2 + cfg.n_stages * cfg.n_trials)

    print(f"\nBuilding healthy Watts-Strogatz network ...")
    G0 = build_initial_network(cfg, int(net_ss.generate_state(1)[0]))
    print(f"  ⟨k⟩={np.mean([d for _, d in G0.degree()]):.2f}  "
          f"|E|={G0.number_of_edges()}")

    # Precompute betweenness ordering on the healthy graph
    edge_order = None
    if cfg.ablation_strategy == "betweenness" and not cfg.recompute_betweenness:
        print("Computing edge-betweenness on healthy graph (one-shot) ...")
        t0 = time.time()
        eb = nx.edge_betweenness_centrality(G0)
        edge_order = sorted(eb, key=eb.get, reverse=True)
        print(f"  done in {time.time()-t0:.1f}s")

    abl_rng = np.random.default_rng(int(abl_ss.generate_state(1)[0]))

    R_global  = np.zeros((cfg.n_stages, cfg.n_trials))
    R_comp    = np.zeros((cfg.n_stages, cfg.n_trials))
    R_global_std = np.zeros((cfg.n_stages, cfg.n_trials))
    R_comp_std   = np.zeros((cfg.n_stages, cfg.n_trials))
    metrics_per_stage = []
    per_stage_per_component = [None] * cfg.n_stages # for storing R_k per component if needed for post-hoc analysis
    full_traces = {}
    networks_per_stage = {}        # store G for raster_stages for visualisation

    t_start = time.time()

    for s in range(cfg.n_stages):
        stage = s + 1
        frac = cfg.ablation_fraction(stage)

        if cfg.ablation_strategy == "betweenness":
            if cfg.recompute_betweenness:
                G, _ = ablate_by_betweenness(G0, frac)   # recompute on G0 each time
            else:
                G, _ = ablate_by_betweenness(G0, frac, edge_order=edge_order)
        elif cfg.ablation_strategy == "random":
            G = ablate_random(G0, frac, abl_rng)
        else:
            raise ValueError(f"Unknown strategy {cfg.ablation_strategy}")

        m = compute_graph_metrics(G)
        metrics_per_stage.append(m)

        print(f"\nStage {stage:2d}/{cfg.n_stages}: ablated {frac*100:.0f}%  "
              f"|E|={m['n_edges']}  CC={m['n_components']}  "
              f"largest={m['largest_cc_size']}  "
              f"clust={m['mean_clustering']:.3f}  "
              f"eff={m['global_efficiency']:.3f}  "
              f"mod={m['modularity']:.3f}")

        if stage in cfg.raster_stages:
            networks_per_stage[stage] = G

        for k in range(cfg.n_trials):
            trial_seed = int(trial_ss[s * cfg.n_trials + k]
                              .generate_state(1)[0])
            need_for_raster = (k == 0 and stage in cfg.raster_stages)
            need_for_save   = (k == 0 and stage in cfg.save_trajectories_at)
            store = need_for_raster or need_for_save

            t0 = time.time()
            res = simulate(G, cfg, seed=trial_seed, store_full=store)
            R_global[s, k]  = res["R_global_mean"]
            R_comp[s, k]    = res["R_component_mean"]
            R_global_std[s, k] = res["R_global_std"]
            R_comp_std[s, k]   = res["R_component_std"]
            # Capture per-component snapshot once per stage (trial 0)
            if k == 0:
                per_stage_per_component[s] = {
                    "sizes":      res["component_sizes"].tolist(),
                    "R_k":        res["R_per_component"].tolist(),
                    "n_isolates": res["n_isolates"],
                }
            print(f"  trial {k+1}/{cfg.n_trials}: "
                  f"R_glob={res['R_global_mean']:.4f}  "
                  f"R_comp={res['R_component_mean']:.4f}  "
                  f"({time.time()-t0:.1f}s)")

            if store:
                full_traces[stage] = (res["V"], res["W"])
            # Persist trajectory to disk if requested
            if need_for_save and cfg.trajectory_outdir:
                traj_dir = Path(cfg.trajectory_outdir)
                traj_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    traj_dir / f"traj_stage{stage:02d}_trial{k}.npz",
                    V=res["V"], W=res["W"],
                    dt=cfg.dt,
                    burn_in_steps=int(cfg.burn_in / cfg.dt),
                    stage=stage, K=cfg.K, sigma=cfg.sigma, a=cfg.a,
                )

    print(f"\nTotal wall-time: {time.time()-t_start:.1f}s")

    return dict(
        R_global=R_global, R_component=R_comp,
        R_global_std=R_global_std, R_component_std=R_comp_std,
        metrics_per_stage=metrics_per_stage,
        full_traces=full_traces,
        networks_per_stage=networks_per_stage,
        G0=G0,
        per_stage_per_component=per_stage_per_component,
    )


# ---- plotting ----
def _detect_spikes(V, dt, threshold=0.5):
    above = V > threshold
    rising = np.diff(above.astype(np.int8), axis=0) == 1
    time_idx, neuron_idx = np.where(rising)
    return neuron_idx, time_idx * dt


def plot_R_ensemble(R_glob, R_comp, cfg, outpath):
    stages = np.arange(1, cfg.n_stages + 1)
    mu_g = R_glob.mean(axis=1);  se_g = R_glob.std(axis=1) / np.sqrt(cfg.n_trials)
    mu_c = R_comp.mean(axis=1);  se_c = R_comp.std(axis=1) / np.sqrt(cfg.n_trials)

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for k in range(cfg.n_trials):
        ax.plot(stages, R_glob[:, k], "-", color="C0", alpha=0.20, lw=0.8)
        ax.plot(stages, R_comp[:, k], "-", color="C3", alpha=0.20, lw=0.8)

    ax.errorbar(stages, mu_g, yerr=se_g, fmt="o-", color="C0",
                 lw=2, ms=7, capsize=4,
                 label=f"R global  (mean ± SE, n={cfg.n_trials})")
    ax.errorbar(stages, mu_c, yerr=se_c, fmt="s-", color="C3",
                 lw=2, ms=7, capsize=4,
                 label=f"R per component (size-weighted)")
    ax.fill_between(stages, mu_g - se_g, mu_g + se_g, alpha=0.15, color="C0")
    ax.fill_between(stages, mu_c - se_c, mu_c + se_c, alpha=0.15, color="C3")

    ax.set_xlabel("Disease stage  (→ more ablation)")
    ax.set_ylabel("Kuramoto order parameter  R")
    ax.set_title(f"Alzheimer's progression — {cfg.ablation_strategy} ablation\n"
                  f"K={cfg.K}, a={cfg.a}, ε={cfg.eps}")
    ax.set_xticks(stages)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_raster(V, stage, cfg, m, outpath):
    n_idx, t_spk = _detect_spikes(V, cfg.dt, threshold=0.5)
    T_steps, N = V.shape
    t_axis = np.arange(T_steps) * cfg.dt

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1]})
    show_max = min(200, N)
    show_mask = n_idx < show_max
    ax[0].scatter(t_spk[show_mask], n_idx[show_mask], s=2, c="black",
                   marker="|", linewidths=0.5)
    ax[0].set_ylim(-1, show_max)
    ax[0].set_ylabel(f"Neuron (first {show_max})", fontsize=16)
    ax[1].plot(t_axis, V.mean(axis=1), lw=0.7, color="C0")
    ax[1].set_xlabel("Time", fontsize=16)
    ax[1].set_ylabel("⟨v⟩(t)", fontsize=16)
    ax[1].grid(True, alpha=0.3)
    # the larger tick font would otherwise drop the top tick
    ax[1].yaxis.set_major_locator(plt.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    for a_ in ax:
        a_.tick_params(axis="both", which="major", labelsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_network(G, stage, cfg, m, pos, outpath):
    """Network visualisation with components in different colours."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Colour nodes by component (largest = blue, rest = warm colours)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    colour_map = {}
    palette = plt.cm.tab20.colors
    for i, comp in enumerate(components):
        if i == 0:
            c = "#1f77b4"      # largest CC: blue
        else:
            c = palette[(i - 1) % len(palette)]
        for node in comp:
            colour_map[node] = c
    node_colours = [colour_map[n] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.4, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=12, node_color=node_colours,
                            linewidths=0, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_graph_metrics(metrics_per_stage, cfg, outpath):
    stages = np.arange(1, cfg.n_stages + 1)
    keys = ["n_edges", "largest_cc_size", "global_efficiency",
            "mean_clustering", "modularity", "avg_path_length_lcc"]
    titles = ["Edge count", "Largest CC size",
              "Global efficiency", "Mean clustering",
              "Modularity (greedy)", "Avg path length (LCC)"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key, title in zip(axes.flat, keys, titles):
        vals = [m[key] for m in metrics_per_stage]
        ax.plot(stages, vals, "o-", lw=1.8, ms=6, color="C0")
        ax.set_xlabel("Stage");  ax.set_ylabel(title)
        ax.set_xticks(stages)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_phase_portrait(cfg, outpath):
    """Single-neuron phase portrait, the same at every stage."""
    rng = np.random.default_rng(42)
    v_star, w_star = fhn_fixed_point(cfg.a, cfg.b, cfg.I_ext)

    v, w = v_star + 0.05, w_star
    n = int(200.0 / cfg.dt)
    V_iso = np.empty(n);  W_iso = np.empty(n)
    sqrt_dt = np.sqrt(cfg.dt)
    for i in range(n):
        dv = v - v**3 / 3.0 - w + cfg.I_ext
        dw = cfg.eps * (v + cfg.a - cfg.b * w)
        v += cfg.dt * dv + cfg.sigma * sqrt_dt * rng.standard_normal()
        w += cfg.dt * dw
        V_iso[i] = v;  W_iso[i] = w

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    t_axis = np.arange(n) * cfg.dt
    ax[0].plot(t_axis, V_iso, lw=0.6, color="C0")
    ax[0].set_xlabel("Time");  ax[0].set_ylabel("v(t)")
    ax[0].grid(True, alpha=0.3)

    v_range = np.linspace(-2.5, 2.5, 300)
    ax[1].plot(v_range, v_range - v_range**3/3.0 + cfg.I_ext, "--", color="C3",
                label="v-nullcline")
    ax[1].plot(v_range, (v_range + cfg.a) / cfg.b, "--", color="C0",
                label="w-nullcline")
    ax[1].plot(V_iso[2000:], W_iso[2000:], lw=0.4, color="C1", alpha=0.7,
                label="trajectory")
    ax[1].plot(v_star, w_star, "ko", ms=8,
                label=f"FP ({v_star:.2f}, {w_star:.2f})")
    ax[1].set_xlim(-2.5, 2.5);  ax[1].set_ylim(-1.5, 2.5)
    ax[1].set_xlabel("v");  ax[1].set_ylabel("w")
    ax[1].legend(loc="upper left", fontsize=8);  ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---- main ----
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ablation", choices=["betweenness", "random"],
                         default=None)
    parser.add_argument("--recompute-betweenness", action="store_true")
    parser.add_argument("--outdir", type=str, default="alzheimers_results")
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.N, cfg.n_trials = 300, 2
        cfg.T, cfg.burn_in = 150.0, 50.0
    if args.trials is not None: cfg.n_trials = args.trials
    if args.stages is not None: cfg.n_stages = args.stages
    if args.N is not None: cfg.N = args.N
    if args.seed is not None: cfg.master_seed = args.seed
    if args.ablation is not None: cfg.ablation_strategy = args.ablation
    if args.recompute_betweenness: cfg.recompute_betweenness = True

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = run_ensemble(cfg)

    print("\nGenerating figures...")
    plot_R_ensemble(results["R_global"], results["R_component"], cfg,
                     outdir / "R_stages_ensemble.png")
    plot_graph_metrics(results["metrics_per_stage"], cfg,
                        outdir / "graph_metrics.png")
    plot_phase_portrait(cfg, outdir / "phase_portrait.png")

    # fixed layout from the healthy graph, so fragmentation shows across stages
    print("Computing layout on healthy graph for visualisation...")
    pos = nx.spring_layout(results["G0"], seed=cfg.master_seed, iterations=50)

    for stage in cfg.raster_stages:
        if stage in results["networks_per_stage"]:
            G = results["networks_per_stage"][stage]
            m = results["metrics_per_stage"][stage - 1]
            plot_network(G, stage, cfg, m, pos,
                          outdir / f"network_stage{stage}.png")
        if stage in results["full_traces"]:
            V, W = results["full_traces"][stage]
            m = results["metrics_per_stage"][stage - 1]
            plot_raster(V, stage, cfg, m,
                         outdir / f"raster_stage{stage}.png")

    np.savez(
        outdir / "results.npz",
        R_global=results["R_global"],
        R_component=results["R_component"],
        R_global_std=results["R_global_std"],
        R_component_std=results["R_component_std"],
        metrics=np.array([json.dumps(m) for m in results["metrics_per_stage"]]),
        config=np.array(json.dumps(asdict(cfg))),
    )

    summary = dict(
        config=asdict(cfg),
        R_global_mean=results["R_global"].mean(axis=1).tolist(),
        R_global_se=(results["R_global"].std(axis=1)
                       / np.sqrt(cfg.n_trials)).tolist(),
        R_component_mean=results["R_component"].mean(axis=1).tolist(),
        R_component_se=(results["R_component"].std(axis=1)
                          / np.sqrt(cfg.n_trials)).tolist(),
        metrics_per_stage=results["metrics_per_stage"],
    )
    with open(outdir / "run_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"RESULTS  ({cfg.ablation_strategy} ablation)")
    print("=" * 64)
    print(f"{'Stage':>5}  {'%abl':>5}  {'|E|':>5}  {'CC':>3}  "
          f"{'R_global':>10}  {'R_component':>11}")
    for s in range(cfg.n_stages):
        m = results["metrics_per_stage"][s]
        Rg = results["R_global"][s].mean()
        Rg_se = results["R_global"][s].std() / np.sqrt(cfg.n_trials)
        Rc = results["R_component"][s].mean()
        Rc_se = results["R_component"][s].std() / np.sqrt(cfg.n_trials)
        print(f"  {s+1:>3}  {cfg.ablation_fraction(s+1)*100:>4.0f}%  "
              f"{m['n_edges']:>5}  {m['n_components']:>3}  "
              f"{Rg:>6.3f}±{Rg_se:.3f}  {Rc:>6.3f}±{Rc_se:.3f}")
    print(f"\n✓ All outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()

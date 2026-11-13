"""
Revision extension for AIP Advances MS #ADV26-AR-04800-TR.

Adds what reviewer points 2 and 3 need and the published code does not have:

  * node-based ablation, in two normalisations (see NODE ABLATION below)
  * a generalised ensemble driver that dispatches on six protocols
  * a light metrics path, because the published metrics call
    greedy_modularity_communities and global_efficiency at every stage, which
    does not scale to the N = 2000 sweep

Nothing in `fhn-alzheimers/` is modified. This module imports from it and the
published code stays byte-identical, so the released repository keeps matching
the submitted paper until the revision is accepted.

NODE ABLATION
Removing a neuron is not the same operation as removing a synapse, and the
difference lands on the denominator of R_g, which is an average over the
population. Two normalisations are therefore implemented and both are run,
rather than one being chosen here:

  excise   the node and its edges are deleted. N shrinks with damage, so R_g is
           an average over survivors. This is neuron death.
  isolate  the node keeps its place in the population but loses every edge. N is
           fixed, so R_g stays directly comparable with the edge-ablation runs.
           The neuron is above the Hopf threshold, so it goes on oscillating and
           drifts out of phase. This is synaptic isolation of a living cell.

Under both, an isolated node is a component of size 1 and is excluded from R_c,
which only aggregates components of size >= 2. That is the published convention
and it is left alone.

COMMON AXIS
A node fraction and an edge fraction are not comparable numbers. Removing a
fraction q of nodes at random removes about 2q - q^2 of the edges, so q = 0.60
already costs about 84% of the edges, and targeted node removal is faster still.
Every stage therefore records the edge fraction it actually achieved, in
`edge_fraction_achieved`, and node and edge protocols are to be plotted against
that measured quantity rather than against their own control parameter.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace as dc_replace
from pathlib import Path

import numpy as np
import networkx as nx

# the published code lives one directory up and is imported, not copied
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fhn_alzheimers import (  # noqa: E402
    Config,
    build_initial_network,
    ablate_by_betweenness,
    ablate_random,
    compute_graph_metrics,
    simulate,
)

PROTOCOLS = (
    "edge_targeted",    # published: highest edge betweenness first
    "edge_random",      # published: uniform random edges
    "node_targeted_excise",
    "node_targeted_isolate",
    "node_random_excise",
    "node_random_isolate",
)

# a node fraction of 0.60 costs roughly the same edges as the published 0.85
# edge fraction; see COMMON AXIS above
DEFAULT_NODE_ABLATION_MAX = 0.60


# ---- node ablation ----
def node_betweenness_order(G: nx.Graph) -> list:
    """Nodes by descending betweenness centrality, computed once on the healthy
    graph. Mirrors the one-shot edge-betweenness convention of the published
    runs, where the ordering is not recomputed as damage accumulates."""
    nb = nx.betweenness_centrality(G)
    return sorted(nb, key=nb.get, reverse=True)


def _apply_node_removal(G: nx.Graph, victims, mode: str) -> nx.Graph:
    G_abl = G.copy()
    if mode == "excise":
        G_abl.remove_nodes_from(victims)
    elif mode == "isolate":
        G_abl.remove_edges_from(list(G_abl.edges(victims)))
    else:
        raise ValueError(f"mode must be excise or isolate, got {mode!r}")
    return G_abl


def ablate_nodes_by_betweenness(G: nx.Graph, fraction: float, mode: str,
                                node_order: list | None = None
                                ) -> tuple[nx.Graph, list]:
    """Remove the highest-betweenness `fraction` of nodes."""
    if node_order is None:
        node_order = node_betweenness_order(G)
    n_remove = int(round(fraction * len(node_order)))
    return _apply_node_removal(G, node_order[:n_remove], mode), node_order


def ablate_nodes_random(G: nx.Graph, fraction: float, mode: str,
                        rng: np.random.Generator) -> nx.Graph:
    """Control: remove `fraction` of nodes uniformly at random."""
    nodes = list(G.nodes())
    n_remove = int(round(fraction * len(nodes)))
    perm = rng.permutation(len(nodes))
    return _apply_node_removal(G, [nodes[i] for i in perm[:n_remove]], mode)


# ---- metrics ----
def light_metrics(G: nx.Graph) -> dict:
    """Structural metrics that are cheap at every N. The published
    compute_graph_metrics also returns modularity, global efficiency and mean
    path length, each of which is superlinear in N and is already reported at
    N = 1000 in the submitted paper, so the sweeps do not recompute them."""
    components = list(nx.connected_components(G))
    n_nodes = G.number_of_nodes()
    largest = max(components, key=len) if components else set()
    sizes = np.array(sorted((len(c) for c in components), reverse=True))
    big = sizes[sizes >= 2]
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(G.number_of_edges()),
        "n_components": int(len(components)),
        "largest_cc_size": int(len(largest)),
        "fragmentation": float(1.0 - len(largest) / n_nodes) if n_nodes else 1.0,
        "n_isolates": int((sizes == 1).sum()),
        # participation ratio of component sizes, the PR of the paper's
        # decorrelated-phase reference level R_g^pred = R_c sqrt(PR)
        "participation_ratio": (float((big**2).sum() / (big.sum() ** 2))
                                if big.size else 0.0),
    }


# ---- generalised single-network ensemble ----
def run_ensemble_ext(cfg: Config, protocol: str, *,
                     node_ablation_max: float = DEFAULT_NODE_ABLATION_MAX,
                     full_metrics: bool = False,
                     verbose: bool = True) -> dict:
    """One network, all stages, `cfg.n_trials` trials per stage.

    Mirrors fhn_alzheimers.run_ensemble, including its SeedSequence discipline,
    but dispatches over all six protocols and records the achieved edge fraction
    at every stage. Trajectories are never stored, the sweeps do not need them.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")

    is_node = protocol.startswith("node_")
    mode = ("excise" if protocol.endswith("_excise")
            else "isolate" if protocol.endswith("_isolate") else None)
    abl_max = node_ablation_max if is_node else cfg.ablation_max

    root = np.random.SeedSequence(cfg.master_seed)
    net_ss, abl_ss, *trial_ss = root.spawn(2 + cfg.n_stages * cfg.n_trials)
    abl_rng = np.random.default_rng(abl_ss)

    G0 = build_initial_network(cfg, int(net_ss.generate_state(1)[0]))
    E0 = G0.number_of_edges()
    if verbose:
        k_mean = np.mean([d for _, d in G0.degree()])
        print(f"  healthy graph: N={G0.number_of_nodes()} |E|={E0} "
              f"<k>={k_mean:.2f}  protocol={protocol}  max={abl_max}")

    # one-shot centrality ordering on the healthy graph, as published
    edge_order = node_order = None
    if protocol == "edge_targeted":
        t0 = time.time()
        eb = nx.edge_betweenness_centrality(G0)
        edge_order = sorted(eb, key=eb.get, reverse=True)
        if verbose:
            print(f"  edge betweenness: {time.time()-t0:.1f}s")
    elif protocol == "node_targeted_excise" or protocol == "node_targeted_isolate":
        t0 = time.time()
        node_order = node_betweenness_order(G0)
        if verbose:
            print(f"  node betweenness: {time.time()-t0:.1f}s")

    n_s, n_t = cfg.n_stages, cfg.n_trials
    R_global = np.zeros((n_s, n_t))
    R_comp = np.zeros((n_s, n_t))
    metrics_per_stage = []
    control_fraction = np.zeros(n_s)
    edge_fraction_achieved = np.zeros(n_s)

    for s in range(n_s):
        stage = s + 1
        alpha = 0.0 if n_s == 1 else (stage - 1) / (n_s - 1)
        frac = alpha * abl_max
        control_fraction[s] = frac

        if protocol == "edge_targeted":
            G, _ = ablate_by_betweenness(G0, frac, edge_order=edge_order)
        elif protocol == "edge_random":
            G = ablate_random(G0, frac, abl_rng)
        elif is_node and protocol.startswith("node_targeted"):
            G, _ = ablate_nodes_by_betweenness(G0, frac, mode,
                                               node_order=node_order)
        else:
            G = ablate_nodes_random(G0, frac, mode, abl_rng)

        m = (compute_graph_metrics(G) if full_metrics else light_metrics(G))
        m["stage"] = stage
        m["control_fraction"] = float(frac)
        m["edge_fraction_achieved"] = float(1.0 - G.number_of_edges() / E0)
        edge_fraction_achieved[s] = m["edge_fraction_achieved"]
        metrics_per_stage.append(m)

        if verbose:
            print(f"  stage {stage:2d}/{n_s}  ctrl={frac*100:5.1f}%  "
                  f"edges_removed={m['edge_fraction_achieved']*100:5.1f}%  "
                  f"N={m['n_nodes']:4d} |E|={m['n_edges']:5d} "
                  f"CC={m['n_components']:4d} LCC={m['largest_cc_size']:4d}",
                  end="", flush=True)

        for k in range(n_t):
            trial_seed = int(trial_ss[s * n_t + k].generate_state(1)[0])
            res = simulate(G, cfg, seed=trial_seed, store_full=False)
            R_global[s, k] = res["R_global_mean"]
            R_comp[s, k] = res["R_component_mean"]

        if verbose:
            print(f"  Rg={R_global[s].mean():.4f} Rc={R_comp[s].mean():.4f}",
                  flush=True)

    return dict(
        R_global=R_global, R_component=R_comp,
        metrics_per_stage=metrics_per_stage,
        control_fraction=control_fraction,
        edge_fraction_achieved=edge_fraction_achieved,
        n_edges_healthy=E0,
    )


# ---- generalised multi-network driver ----
def run_multi_network_ext(cfg: Config, n_networks: int, protocol: str,
                          **kw) -> dict:
    """`n_networks` independent graphs, each with its own seed stream spawned
    from cfg.master_seed. Mirrors fhn_robustness.run_multi_network."""
    root = np.random.SeedSequence(cfg.master_seed)
    net_seedseqs = root.spawn(n_networks)

    Rg, Rc, mets, ctrl, eachieved = [], [], [], [], []
    for i, net_ss in enumerate(net_seedseqs):
        net_master = int(net_ss.generate_state(1)[0])
        print(f"\n--- network {i+1}/{n_networks}  (seed={net_master}) ---",
              flush=True)
        res = run_ensemble_ext(dc_replace(cfg, master_seed=net_master),
                               protocol, **kw)
        Rg.append(res["R_global"])
        Rc.append(res["R_component"])
        mets.append(res["metrics_per_stage"])
        ctrl.append(res["control_fraction"])
        eachieved.append(res["edge_fraction_achieved"])

    return dict(
        R_global=np.stack(Rg),            # (n_net, n_stages, n_trials)
        R_component=np.stack(Rc),
        metrics_per_network=mets,
        control_fraction=np.stack(ctrl),
        edge_fraction_achieved=np.stack(eachieved),
        protocol=protocol,
    )


# ---- output ----
def save_condition(outdir: Path, label: str, cfg: Config, results: dict,
                   extra: dict | None = None) -> Path:
    """One directory per condition, holding the arrays, the per-stage metrics and
    a summary with the full config so any run can be reproduced from disk."""
    d = Path(outdir) / label
    d.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        d / "data.npz",
        R_global=results["R_global"],
        R_component=results["R_component"],
        control_fraction=results["control_fraction"],
        edge_fraction_achieved=results["edge_fraction_achieved"],
    )
    (d / "metrics_per_network.json").write_text(
        json.dumps(results["metrics_per_network"], indent=2))

    Rg, Rc = results["R_global"], results["R_component"]
    # mean over trials first, then over networks, so the SD is across networks
    Rg_net, Rc_net = Rg.mean(axis=2), Rc.mean(axis=2)
    summary = {
        "label": label,
        "protocol": results["protocol"],
        "config": asdict(cfg),
        "n_networks": int(Rg.shape[0]),
        "n_stages": int(Rg.shape[1]),
        "n_trials": int(Rg.shape[2]),
        "edge_fraction_achieved_mean": results["edge_fraction_achieved"]
                                        .mean(axis=0).tolist(),
        "R_global_mean": Rg_net.mean(axis=0).tolist(),
        "R_global_sd_across_networks": Rg_net.std(axis=0, ddof=1).tolist()
                                        if Rg.shape[0] > 1 else None,
        "R_component_mean": Rc_net.mean(axis=0).tolist(),
        "R_component_sd_across_networks": Rc_net.std(axis=0, ddof=1).tolist()
                                            if Rg.shape[0] > 1 else None,
        "R_global_drop_pct": float(
            100.0 * (1.0 - Rg_net.mean(axis=0)[-1] / Rg_net.mean(axis=0)[0])),
        "R_component_change_pct": float(
            100.0 * (Rc_net.mean(axis=0)[-1] / Rc_net.mean(axis=0)[0] - 1.0)),
    }
    if extra:
        summary.update(extra)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    return d


def canonical_config(**overrides) -> Config:
    """Config defaults are already the canonical parameter set of the paper,
    (K, sigma, a) = (0.35, 0.05, 0.65) with eps = 0.06, T = 400, burn-in 100,
    dt = 0.01. Trajectory saving is switched off for every sweep."""
    cfg = Config(**overrides)
    cfg.save_trajectories_at = ()
    cfg.raster_stages = ()
    return cfg

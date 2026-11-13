"""
sim05 — node-based against edge-based attack.      Reviewer point 3 (R1-3).

"How would the results differ if the damage were applied randomly to neurons or
connections? Although a random edge-removal control is presented, it does not
fully address the possible differences between targeted neuronal damage and
random neuronal damage. The distinction between node-based and edge-based
attacks should be made explicit."

The published code cannot do this. ablate_by_betweenness and ablate_random both
call remove_edges_from, and there is no node operation anywhere in the
repository, so the random control already in the paper is a random EDGE control
and does not answer the question. fhn_ext adds the node operations.

Six protocols on the same axes, a full 2x2 of {targeted, random} x {node, edge}
with the node arm run under both normalisations:

  edge_targeted           published, highest edge betweenness first
  edge_random             published control, uniform random edges
  node_targeted_excise    highest node betweenness, neuron deleted
  node_targeted_isolate   highest node betweenness, neuron kept but decoupled
  node_random_excise      uniform random neurons, deleted
  node_random_isolate     uniform random neurons, kept but decoupled

WHY BOTH NORMALISATIONS ARE RUN
R_g is an average over the population, so deleting neurons changes its
denominator and node-attack R_g is not directly comparable with edge-attack R_g.
Rather than pick one convention and hope it was the right one, both are
measured. A smoke test already shows the two diverge, with isolate pushing R_g
lower because a decoupled neuron stays above the Hopf threshold, goes on
oscillating and drifts out of phase, while excise leaves a denser and better
synchronised remainder. Which one the paper should lead with is an author
decision, and this run supplies the evidence for it either way.

AXES
Node fractions and edge fractions are different quantities. Removing a fraction
q of nodes at random removes about 2q - q^2 of the edges, so the node arms run
to a node fraction of 0.60, which costs roughly the 85% of edges that the
published edge schedule removes. Every stage records the edge fraction it
actually achieved, and the comparison figure must be drawn against that measured
axis rather than against each protocol's own control parameter.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import fhn_ext as X

SIM = "sim05_node_ablation"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", type=int, default=10)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--stages", type=int, default=10)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--k-nbr", type=int, default=8)
    p.add_argument("--protocols", type=str, nargs="+",
                   default=list(X.PROTOCOLS))
    p.add_argument("--node-ablation-max", type=float,
                   default=X.DEFAULT_NODE_ABLATION_MAX)
    p.add_argument("--outdir", type=str, default=f"runs/{SIM}")
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    if a.quick:
        a.networks, a.trials, a.stages, a.N = 2, 1, 3, 100

    for proto in a.protocols:
        if proto not in X.PROTOCOLS:
            raise SystemExit(f"unknown protocol {proto!r}, "
                             f"choose from {X.PROTOCOLS}")

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    index = []

    for proto in a.protocols:
        cfg = X.canonical_config(N=a.N, k_nbr=a.k_nbr, n_stages=a.stages,
                                 n_trials=a.trials, master_seed=a.seed)
        if a.quick:
            cfg.T, cfg.burn_in = 20.0, 5.0

        print(f"\n{'='*70}\n{SIM}  {proto}  N={a.N} <k>={a.k_nbr}  "
              f"networks={a.networks}  trials={a.trials}\n{'='*70}", flush=True)
        t0 = time.time()
        res = X.run_multi_network_ext(cfg, a.networks, proto,
                                      node_ablation_max=a.node_ablation_max)
        d = X.save_condition(
            out, proto, cfg, res,
            extra={"sim": SIM, "N": a.N, "k_nbr": a.k_nbr,
                   "node_ablation_max": a.node_ablation_max,
                   "is_node_protocol": proto.startswith("node_"),
                   "normalisation": ("excise" if proto.endswith("_excise")
                                     else "isolate" if proto.endswith("_isolate")
                                     else "n/a"),
                   "wall_seconds": round(time.time() - t0, 1)})
        s = json.loads((d / "summary.json").read_text())
        print(f"\n{proto}: R_g {s['R_global_mean'][0]:.4f} -> "
              f"{s['R_global_mean'][-1]:.4f} ({s['R_global_drop_pct']:.1f}% drop)   "
              f"R_c {s['R_component_mean'][0]:.4f} -> "
              f"{s['R_component_mean'][-1]:.4f} "
              f"({s['R_component_change_pct']:+.2f}%)   "
              f"edges removed by last stage "
              f"{s['edge_fraction_achieved_mean'][-1]*100:.1f}%   "
              f"[{time.time()-t0:.0f}s]", flush=True)
        index.append({
            "protocol": proto,
            "normalisation": s["normalisation"],
            "R_global_drop_pct": s["R_global_drop_pct"],
            "R_component_change_pct": s["R_component_change_pct"],
            "edge_fraction_final": s["edge_fraction_achieved_mean"][-1],
            "R_global_final": s["R_global_mean"][-1],
            "R_component_final": s["R_component_mean"][-1],
            "wall_seconds": s["wall_seconds"],
        })

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\n{SIM} done in {(time.time()-t_all)/60:.1f} min -> {out}")

    print(f"\n{'protocol':>24} {'norm':>8} {'edges rm':>9} "
          f"{'R_g drop %':>11} {'R_c chg %':>10}")
    for r in index:
        print(f"{r['protocol']:>24} {r['normalisation']:>8} "
              f"{r['edge_fraction_final']*100:>8.1f}% "
              f"{r['R_global_drop_pct']:>11.1f} "
              f"{r['R_component_change_pct']:>10.2f}")


if __name__ == "__main__":
    main()

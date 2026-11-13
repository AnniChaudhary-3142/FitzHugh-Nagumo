"""
sim02 — average-degree sweep.          Reviewer point 2 (R1-2), part 2 of 4.

"Most of the numerical analysis is performed for a fixed network size, N=1000,
and a fixed average degree."

The published runs hold <k> = 8 everywhere. Average degree is the second half of
R1-2 and is easy to miss on a first reading of the report. It matters for the
same reason N does: in a Watts-Strogatz graph the rewired long-range shortcuts
carry the highest betweenness, and <k> sets how many of them there are, so it
controls how much of the network a betweenness-targeted attack can reach before
the graph comes apart.

Edge-targeted ablation at N = 1000, canonical (K, sigma, a) = (0.35, 0.05, 0.65),
ten stages, ten realisations, three trials.

A caution for interpretation. K is the coupling strength per neighbour and the
drift term divides by degree, so changing <k> does not change the total drive a
neuron receives. The sweep therefore isolates topology rather than confounding
it with coupling. Worth stating explicitly in the paper, because a reader will
reasonably wonder.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import fhn_ext as X

SIM = "sim02_degree_sweep"
DEGREES = (4, 6, 8, 12)   # watts_strogatz_graph needs k even and >= 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", type=int, default=10)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--stages", type=int, default=10)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    p.add_argument("--outdir", type=str, default=f"runs/{SIM}")
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    if a.quick:
        a.networks, a.trials, a.stages, a.N = 2, 1, 3, 100
        a.degrees = [4, 8]

    for k in a.degrees:
        if k % 2 or k < 2:
            raise SystemExit(f"watts_strogatz_graph needs k even and >= 2, got {k}")

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    index = []

    for k in a.degrees:
        cfg = X.canonical_config(N=a.N, k_nbr=k, n_stages=a.stages,
                                 n_trials=a.trials, master_seed=a.seed)
        if a.quick:
            cfg.T, cfg.burn_in = 20.0, 5.0

        label = f"k{k}"
        print(f"\n{'='*70}\n{SIM}  {label}  "
              f"N={a.N}  networks={a.networks}  trials={a.trials}\n{'='*70}",
              flush=True)
        t0 = time.time()
        res = X.run_multi_network_ext(cfg, a.networks, "edge_targeted")
        d = X.save_condition(out, label, cfg, res,
                             extra={"sim": SIM, "N": a.N, "k_nbr": k,
                                    "wall_seconds": round(time.time() - t0, 1)})
        s = json.loads((d / "summary.json").read_text())
        print(f"\n{label}: R_g {s['R_global_mean'][0]:.4f} -> "
              f"{s['R_global_mean'][-1]:.4f} ({s['R_global_drop_pct']:.1f}% drop)   "
              f"R_c {s['R_component_mean'][0]:.4f} -> "
              f"{s['R_component_mean'][-1]:.4f} "
              f"({s['R_component_change_pct']:+.2f}%)   "
              f"[{time.time()-t0:.0f}s]", flush=True)
        index.append({kk: s[kk] for kk in
                      ("label", "N", "k_nbr", "R_global_drop_pct",
                       "R_component_change_pct", "wall_seconds")})

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\n{SIM} done in {(time.time()-t_all)/60:.1f} min -> {out}")

    print(f"\n{'<k>':>5} {'R_g drop %':>12} {'R_c change %':>14}")
    for r in index:
        print(f"{r['k_nbr']:>5} {r['R_global_drop_pct']:>12.1f} "
              f"{r['R_component_change_pct']:>14.2f}")


if __name__ == "__main__":
    main()

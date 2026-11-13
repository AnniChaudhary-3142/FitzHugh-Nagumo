"""
sim01 — network-size sweep.            Reviewer point 2 (R1-2), part 1 of 4.

"How would the results change for larger or smaller networks?"

Edge-targeted ablation, the published protocol, at five network sizes with the
average degree held at the published <k> = 8. Canonical (K, sigma, a) =
(0.35, 0.05, 0.65). Ten stages spanning 0 to 85% edge removal, ten independent
Watts-Strogatz realisations, three trials each, which is exactly the ensemble
behind the submitted paper's canonical figure.

N = 1000 is included deliberately and is not redundant: it must reproduce the
published numbers, R_g falling 47% from 0.9685 to 0.5084 while R_c rises 0.35%.
If it does not, the extension module is wrong and nothing else here is
trustworthy. Check it first.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import fhn_ext as X

SIM = "sim01_size_sweep"
SIZES = (250, 500, 1000, 2000, 4000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", type=int, default=10)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--stages", type=int, default=10)
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--outdir", type=str, default=f"runs/{SIM}")
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke test, not for results")
    a = p.parse_args()

    if a.quick:
        a.networks, a.trials, a.stages = 2, 1, 3
        a.sizes = [100, 200]

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    index = []

    for N in a.sizes:
        cfg = X.canonical_config(N=N, k_nbr=8, n_stages=a.stages,
                                 n_trials=a.trials, master_seed=a.seed)
        if a.quick:
            cfg.T, cfg.burn_in = 20.0, 5.0

        label = f"N{N}"
        print(f"\n{'='*70}\n{SIM}  {label}  "
              f"<k>=8  networks={a.networks}  trials={a.trials}\n{'='*70}",
              flush=True)
        t0 = time.time()
        res = X.run_multi_network_ext(cfg, a.networks, "edge_targeted")
        d = X.save_condition(out, label, cfg, res,
                             extra={"sim": SIM, "N": N, "k_nbr": 8,
                                    "wall_seconds": round(time.time() - t0, 1)})
        s = json.loads((d / "summary.json").read_text())
        print(f"\n{label}: R_g {s['R_global_mean'][0]:.4f} -> "
              f"{s['R_global_mean'][-1]:.4f} ({s['R_global_drop_pct']:.1f}% drop)   "
              f"R_c {s['R_component_mean'][0]:.4f} -> "
              f"{s['R_component_mean'][-1]:.4f} "
              f"({s['R_component_change_pct']:+.2f}%)   "
              f"[{time.time()-t0:.0f}s]", flush=True)
        index.append({k: s[k] for k in
                      ("label", "N", "k_nbr", "R_global_drop_pct",
                       "R_component_change_pct", "wall_seconds")})

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\n{SIM} done in {(time.time()-t_all)/60:.1f} min -> {out}")

    print(f"\n{'N':>6} {'R_g drop %':>12} {'R_c change %':>14}")
    for r in index:
        print(f"{r['N']:>6} {r['R_global_drop_pct']:>12.1f} "
              f"{r['R_component_change_pct']:>14.2f}")


if __name__ == "__main__":
    main()

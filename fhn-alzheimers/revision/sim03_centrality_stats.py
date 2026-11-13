"""
sim03 — betweenness distribution against N and <k>.
                                       Reviewer point 2 (R1-2), part 3 of 4.

"Since the removal protocol is based on centrality, the distribution of
centrality values and its dependence on network size may also influence the
damage progression."

This is the sharpest sentence in the whole report and it needs no dynamics at
all, only graph construction, so it is by far the cheapest substantive answer
available. It is also the one that reframes the other results: if the
betweenness distribution is not size-invariant then "remove the top 85% of
edges by betweenness" is simply not the same intervention at N = 250 as at
N = 4000, and any N-dependence found in sim01 and sim04 has its explanation
here rather than in the dynamics.

Reports, for each (N, <k>) cell over many realisations:

  * the edge- and node-betweenness distributions, normalised and raw
  * Gini coefficient and the top-1% and top-10% concentration shares, which say
    how unequally betweenness is spread and therefore how much damage a targeted
    attack concentrates
  * the share of total betweenness carried by the edges removed at each of the
    ten published ablation stages, which is the protocol's actual bite
  * mean and max betweenness, and the coefficient of variation

Scaling note for interpretation. networkx normalises betweenness by the number
of node pairs, so normalised values are comparable across N by construction.
The raw, unnormalised values are also reported, because the quantity the attack
actually sorts on is the raw one and its spread is what changes with N.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx

import fhn_ext as X

SIM = "sim03_centrality_stats"
SIZES = (250, 500, 1000, 2000, 4000)
DEGREES = (4, 6, 8, 12)
STAGE_FRACTIONS = tuple(i / 9.0 * 0.85 for i in range(10))   # published schedule


def gini(x: np.ndarray) -> float:
    """Standard Gini coefficient. 0 is perfectly even, 1 is all mass on one edge."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return float("nan")
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def describe(vals: np.ndarray, n_pairs: float) -> dict:
    """vals are networkx-normalised betweenness values."""
    v = np.sort(np.asarray(vals, dtype=float))[::-1]
    total = v.sum()
    m = v.size
    raw = v * n_pairs                     # undo the pair normalisation
    out = {
        "count": int(m),
        "mean_norm": float(v.mean()),
        "max_norm": float(v[0]),
        "cv": float(v.std(ddof=1) / v.mean()) if m > 1 and v.mean() > 0 else None,
        "gini": gini(v),
        "mean_raw": float(raw.mean()),
        "max_raw": float(raw[0]),
        "top1pct_share": float(v[: max(1, m // 100)].sum() / total) if total else None,
        "top10pct_share": float(v[: max(1, m // 10)].sum() / total) if total else None,
        # how much of all betweenness the attack has taken by each stage
        "share_removed_by_stage": [
            float(v[: int(round(f * m))].sum() / total) if total else None
            for f in STAGE_FRACTIONS
        ],
    }
    return out


def cell(N: int, k: int, n_real: int, seed: int) -> dict:
    eb_stats, nb_stats = [], []
    n_pairs_nodes = N * (N - 1) / 2.0
    for r in range(n_real):
        G = nx.watts_strogatz_graph(N, k, 0.1, seed=seed + r)
        eb = np.fromiter(nx.edge_betweenness_centrality(G).values(), dtype=float)
        nb = np.fromiter(nx.betweenness_centrality(G).values(), dtype=float)
        eb_stats.append(describe(eb, n_pairs_nodes))
        nb_stats.append(describe(nb, n_pairs_nodes))

    def agg(rows):
        keys = [kk for kk, vv in rows[0].items() if isinstance(vv, (int, float))
                and vv is not None]
        o = {kk: float(np.mean([r[kk] for r in rows])) for kk in keys}
        o.update({kk + "_sd": float(np.std([r[kk] for r in rows], ddof=1))
                  for kk in keys if len(rows) > 1})
        o["share_removed_by_stage"] = np.mean(
            [r["share_removed_by_stage"] for r in rows], axis=0).tolist()
        return o

    return {"N": N, "k_nbr": k, "n_realisations": n_real,
            "edge_betweenness": agg(eb_stats),
            "node_betweenness": agg(nb_stats)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--realisations", type=int, default=20)
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    p.add_argument("--outdir", type=str, default=f"runs/{SIM}")
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    if a.quick:
        a.realisations, a.sizes, a.degrees = 2, [100, 200], [4, 8]

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    cells = []

    # size sweep at the published <k> = 8
    for N in a.sizes:
        k = 8
        t0 = time.time()
        c = cell(N, k, a.realisations, a.seed)
        c["arm"] = "size"
        cells.append(c)
        e = c["edge_betweenness"]
        print(f"size  N={N:5d} k={k:3d}  gini={e['gini']:.3f}  "
              f"cv={e['cv']:.3f}  top10%={e['top10pct_share']:.3f}  "
              f"max_raw={e['max_raw']:.1f}  [{time.time()-t0:.0f}s]", flush=True)

    # degree sweep at the published N = 1000
    N_fix = 1000 if not a.quick else 200
    for k in a.degrees:
        t0 = time.time()
        c = cell(N_fix, k, a.realisations, a.seed)
        c["arm"] = "degree"
        cells.append(c)
        e = c["edge_betweenness"]
        print(f"deg   N={N_fix:5d} k={k:3d}  gini={e['gini']:.3f}  "
              f"cv={e['cv']:.3f}  top10%={e['top10pct_share']:.3f}  "
              f"max_raw={e['max_raw']:.1f}  [{time.time()-t0:.0f}s]", flush=True)

    (out / "centrality_stats.json").write_text(json.dumps(cells, indent=2))
    print(f"\n{SIM} done in {(time.time()-t_all)/60:.1f} min -> {out}")

    print(f"\n{'arm':>6} {'N':>6} {'<k>':>4} {'gini':>7} {'CV':>7} "
          f"{'top1%':>7} {'top10%':>7}")
    for c in cells:
        e = c["edge_betweenness"]
        print(f"{c['arm']:>6} {c['N']:>6} {c['k_nbr']:>4} {e['gini']:>7.3f} "
              f"{e['cv']:>7.3f} {e['top1pct_share']:>7.3f} "
              f"{e['top10pct_share']:>7.3f}")


if __name__ == "__main__":
    main()

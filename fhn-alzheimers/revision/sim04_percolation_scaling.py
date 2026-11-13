"""
sim04 — critical fraction f_c against N and <k>.
                                       Reviewer point 2 (R1-2), part 4 of 4.

"would the critical fraction of removed connections associated with network
fragmentation and the subsequent decrease in global synchronization remain
approximately the same?"

The submitted paper reports f_c ~ 0.30 for betweenness-targeted attack at
N = 1000 and <k> = 8, and uses it in the abstract, in Section 7.1 item 5 and in
the conclusion. It is measured at one size and one degree. This script measures
it across both, structurally, at fine resolution and with no dynamics, so it is
cheap.

f_c is located three ways, because "the critical fraction" is not a single
unambiguous quantity on a finite graph and the paper should say which
definition it uses:

  lcc_half     the fraction at which the largest component first falls below
               N/2. This is the published convention, visible in
               replot_figures.plot_percolation_diagram as "f_c (LCC = N/2)".
  susc_peak    the fraction maximising the mean finite-cluster size excluding
               the largest component, the standard percolation susceptibility.
               This is the definition that converges to the true threshold as
               N grows, so it is the one to quote if f_c turns out to drift.
  lcc_frac10   the fraction at which the largest component first falls below
               10% of N, a deep-fragmentation marker.

Both attack families are measured, since R1-3 asks for the node-versus-edge
distinction to be made explicit and f_c is where that distinction is sharpest.
Node protocols are reported against the edge fraction they actually achieved as
well as against their own node fraction, because the two axes are not
interchangeable.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx

import fhn_ext as X

SIM = "sim04_percolation_scaling"
SIZES = (250, 500, 1000, 2000, 4000)
DEGREES = (4, 6, 8, 12)
ARMS = ("edge_targeted", "edge_random",
        "node_targeted_excise", "node_random_excise")


def sweep_one(G0: nx.Graph, arm: str, fracs: np.ndarray,
              rng: np.random.Generator) -> dict:
    """Structural sweep over removal fractions on one graph. No dynamics."""
    N0 = G0.number_of_nodes()
    E0 = G0.number_of_edges()

    order = None
    if arm == "edge_targeted":
        eb = nx.edge_betweenness_centrality(G0)
        order = sorted(eb, key=eb.get, reverse=True)
    elif arm == "node_targeted_excise":
        order = X.node_betweenness_order(G0)

    lcc, susc, ncomp, edge_frac = [], [], [], []
    for f in fracs:
        if arm == "edge_targeted":
            G, _ = X.ablate_by_betweenness(G0, f, edge_order=order)
        elif arm == "edge_random":
            G = X.ablate_random(G0, f, rng)
        elif arm == "node_targeted_excise":
            G, _ = X.ablate_nodes_by_betweenness(G0, f, "excise",
                                                 node_order=order)
        else:
            G = X.ablate_nodes_random(G0, f, "excise", rng)

        sizes = np.array(sorted((len(c) for c in nx.connected_components(G)),
                                reverse=True))
        # LCC as a fraction of the ORIGINAL N, so node and edge arms share a scale
        lcc.append(float(sizes[0] / N0) if sizes.size else 0.0)
        rest = sizes[1:]
        susc.append(float((rest**2).sum() / rest.sum()) if rest.sum() else 0.0)
        ncomp.append(int(sizes.size))
        edge_frac.append(float(1.0 - G.number_of_edges() / E0))

    return {"lcc_frac": lcc, "susceptibility": susc,
            "n_components": ncomp, "edge_fraction_achieved": edge_frac}


def locate_fc(fracs: np.ndarray, lcc: np.ndarray, susc: np.ndarray) -> dict:
    def first_below(thr):
        idx = np.nonzero(lcc < thr)[0]
        return float(fracs[idx[0]]) if idx.size else None
    return {
        "lcc_half": first_below(0.5),
        "lcc_frac10": first_below(0.10),
        "susc_peak": float(fracs[int(np.argmax(susc))]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", type=int, default=20)
    p.add_argument("--n-fracs", type=int, default=60)
    p.add_argument("--f-max", type=float, default=0.95)
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    p.add_argument("--arms", type=str, nargs="+", default=list(ARMS))
    p.add_argument("--outdir", type=str, default=f"runs/{SIM}")
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    if a.quick:
        a.networks, a.n_fracs = 3, 15
        a.sizes, a.degrees = [100, 200], [4, 8]

    fracs = np.linspace(0.0, a.f_max, a.n_fracs)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    rows = []

    N_fix = 1000 if not a.quick else 200
    _cells = ([("size", N, 8) for N in a.sizes]
              + [("degree", N_fix, k) for k in a.degrees])
    # (N_fix, 8) sits in both arms; run it once and label it as the shared baseline
    cells, seen = [], set()
    for arm, N, k in _cells:
        if (N, k) in seen:
            continue
        seen.add((N, k))
        cells.append(("both" if (N, k) == (N_fix, 8) else arm, N, k))

    for arm in a.arms:
        for armlabel, N, k in cells:
            t0 = time.time()
            per_net = []
            for r in range(a.networks):
                G0 = nx.watts_strogatz_graph(N, k, 0.1, seed=a.seed + r)
                rng = np.random.default_rng(a.seed + 7919 * r)
                per_net.append(sweep_one(G0, arm, fracs, rng))

            lcc = np.array([d["lcc_frac"] for d in per_net])
            susc = np.array([d["susceptibility"] for d in per_net])
            ef = np.array([d["edge_fraction_achieved"] for d in per_net])

            fc = locate_fc(fracs, lcc.mean(axis=0), susc.mean(axis=0))
            # the same thresholds expressed on the achieved-edge-fraction axis
            fc_edge = {}
            for key, val in fc.items():
                if val is None:
                    fc_edge[key] = None
                else:
                    j = int(np.argmin(np.abs(fracs - val)))
                    fc_edge[key] = float(ef.mean(axis=0)[j])

            row = {"sim": SIM, "arm": arm, "sweep": armlabel, "N": N,
                   "k_nbr": k, "n_networks": a.networks,
                   "fracs": fracs.tolist(),
                   "lcc_frac_mean": lcc.mean(axis=0).tolist(),
                   "lcc_frac_sd": lcc.std(axis=0, ddof=1).tolist(),
                   "susceptibility_mean": susc.mean(axis=0).tolist(),
                   "edge_fraction_achieved_mean": ef.mean(axis=0).tolist(),
                   "f_c_control_axis": fc,
                   "f_c_edge_axis": fc_edge,
                   "wall_seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{arm:22s} {armlabel:6s} N={N:5d} k={k:3d}  "
                  f"f_c(LCC=N/2)={fc['lcc_half']}  "
                  f"f_c(susc)={fc['susc_peak']:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    (out / "percolation_scaling.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{SIM} done in {(time.time()-t_all)/60:.1f} min -> {out}")

    print(f"\n{'arm':>22} {'N':>6} {'<k>':>4} {'fc_half':>9} {'fc_susc':>9} "
          f"{'fc_half(edge axis)':>19}")
    for r in rows:
        h = r["f_c_control_axis"]["lcc_half"]
        he = r["f_c_edge_axis"]["lcc_half"]
        print(f"{r['arm']:>22} {r['N']:>6} {r['k_nbr']:>4} "
              f"{(f'{h:.3f}' if h is not None else 'n/a'):>9} "
              f"{r['f_c_control_axis']['susc_peak']:>9.3f} "
              f"{(f'{he:.3f}' if he is not None else 'n/a'):>19}")


if __name__ == "__main__":
    main()

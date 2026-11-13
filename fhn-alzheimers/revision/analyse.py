"""
Collects the sim01-sim05 outputs into the tables the revision needs. Reads
runs/ only and simulates nothing, so it is safe to run repeatedly while the
campaigns are still going. Conditions that have not finished are reported as
pending rather than silently skipped.

Two things it does that the raw summaries do not.

1. VALIDATION. sim01's N = 1000 cell must reproduce the submitted paper, R_g
   falling 47% from 0.9685 to 0.5084 and R_c rising 0.35%. This is checked
   first and loudly, because if the extension module is wrong then nothing else
   in this folder means anything.

2. f_c ON THE PUBLISHED CONVENTION. sim04 stores the full LCC curves, so f_c
   can be re-estimated after the fact. The published fhn_percolation.py
   interpolates between the two grid points that bracket LCC = N/2, while
   sim04's own locate_fc takes the first grid point below it. On the published
   data those differ by +0.0204, so comparing sim04's raw number against the
   paper's 0.321 would be comparing two different estimators. This recomputes
   on the published convention from the stored curves.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"

# submitted paper, canonical N = 1000, <k> = 8, 400-unit window
PUBLISHED = {"Rg_stage1": 0.9685, "Rg_stage10": 0.5084,
             "Rg_drop_pct": 47.0, "Rc_change_pct": 0.35,
             "fc_LCC_half": 0.3210, "fc_steepest": 0.2793}
# the published f_c rests on 2 network realisations whose individual values were
# 0.2992 and 0.3401, so treat it as +-0.02 at best
PUBLISHED_FC_NETWORKS = (0.2992, 0.3401)


def load(sim: str, name: str) -> dict | None:
    p = RUNS / sim / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def fc_interpolated(fracs, rel) -> float | None:
    """Published convention: linear interpolation across the LCC = N/2 crossing."""
    fracs, rel = np.asarray(fracs), np.asarray(rel)
    above, below = np.nonzero(rel >= 0.5)[0], np.nonzero(rel < 0.5)[0]
    if not above.size or not below.size:
        return None
    i1, i2 = above[-1], below[0]
    if i2 - i1 != 1:
        return None
    return float(fracs[i1] + (fracs[i2] - fracs[i1])
                 * (0.5 - rel[i1]) / (rel[i2] - rel[i1]))


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def validate():
    hdr("VALIDATION — sim01 N=1000 must reproduce the submitted paper")
    # sim05's edge_targeted cell is the same configuration (N=1000, <k>=8,
    # 10 networks, 3 trials, canonical parameters), so either validates.
    src = "sim01_size_sweep/N1000"
    s = load("sim01_size_sweep", "N1000")
    if s is None:
        s = load("sim05_node_ablation", "edge_targeted")
        src = "sim05_node_ablation/edge_targeted"
    if s is None:
        print("  pending")
        return
    print(f"  source: {src}")
    rg1, rg10 = s["R_global_mean"][0], s["R_global_mean"][-1]
    rows = [("R_g stage 1", rg1, PUBLISHED["Rg_stage1"], 0.005),
            ("R_g stage 10", rg10, PUBLISHED["Rg_stage10"], 0.02),
            ("R_g drop %", s["R_global_drop_pct"], PUBLISHED["Rg_drop_pct"], 2.0),
            ("R_c change %", s["R_component_change_pct"],
             PUBLISHED["Rc_change_pct"], 0.5)]
    ok = True
    print(f"  {'quantity':>14} {'this run':>10} {'published':>10} "
          f"{'delta':>9} {'tol':>7}  verdict")
    for name, got, exp, tol in rows:
        good = abs(got - exp) <= tol
        ok &= good
        print(f"  {name:>14} {got:>10.4f} {exp:>10.4f} {got-exp:>+9.4f} "
              f"{tol:>7.3f}  {'ok' if good else 'MISMATCH'}")
    print(f"\n  {'PASS, extension reproduces the paper' if ok else 'FAIL, do not trust the other conditions'}")


def sweep_table(sim, key, keyname):
    s_all = []
    d = RUNS / sim
    if not d.exists():
        print("  pending")
        return
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and (sub / "summary.json").exists():
            s_all.append(json.loads((sub / "summary.json").read_text()))
    if not s_all:
        print("  pending")
        return
    s_all.sort(key=lambda s: s[key])
    print(f"  {keyname:>6} {'R_g s1':>8} {'R_g s10':>9} {'drop %':>8} "
          f"{'R_c s1':>8} {'R_c s10':>9} {'chg %':>7} {'SD net s10':>11}")
    for s in s_all:
        sd = s["R_global_sd_across_networks"]
        print(f"  {s[key]:>6} {s['R_global_mean'][0]:>8.4f} "
              f"{s['R_global_mean'][-1]:>9.4f} {s['R_global_drop_pct']:>8.1f} "
              f"{s['R_component_mean'][0]:>8.4f} "
              f"{s['R_component_mean'][-1]:>9.4f} "
              f"{s['R_component_change_pct']:>7.2f} "
              f"{(sd[-1] if sd else float('nan')):>11.4f}")


def centrality():
    hdr("sim03 — betweenness concentration (R1-2, the reviewer's third sentence)")
    p = RUNS / "sim03_centrality_stats" / "centrality_stats.json"
    if not p.exists():
        print("  pending")
        return
    cells = json.loads(p.read_text())
    print("  Is the betweenness distribution size-invariant? If not, "
          "'remove the top\n  85% by betweenness' is a different intervention "
          "at each N.\n")
    print(f"  {'arm':>6} {'N':>6} {'<k>':>4} {'gini':>7} {'CV':>7} "
          f"{'top1%':>7} {'top10%':>7} {'max_raw':>10} {'share@st4':>10}")
    for c in cells:
        e = c["edge_betweenness"]
        st4 = e["share_removed_by_stage"][3]
        print(f"  {c['arm']:>6} {c['N']:>6} {c['k_nbr']:>4} {e['gini']:>7.3f} "
              f"{e['cv']:>7.3f} {e['top1pct_share']:>7.3f} "
              f"{e['top10pct_share']:>7.3f} {e['max_raw']:>10.1f} "
              f"{st4:>10.3f}")
    print("\n  share@st4 is the fraction of ALL betweenness already removed by "
          "stage 4\n  (28.3% of edges). Comparable across N means the attack "
          "bites equally hard.")


def percolation():
    hdr("sim04 — f_c against N and <k> (R1-2)")
    p = RUNS / "sim04_percolation_scaling" / "percolation_scaling.json"
    if not p.exists():
        print("  pending")
        return
    rows = json.loads(p.read_text())
    print(f"  published baseline: f_c = {PUBLISHED['fc_LCC_half']:.4f} "
          f"(LCC=N/2, interpolated, 2 networks: "
          f"{PUBLISHED_FC_NETWORKS[0]:.4f} and {PUBLISHED_FC_NETWORKS[1]:.4f})\n")
    print(f"  {'arm':>22} {'N':>6} {'<k>':>4} {'fc interp':>10} "
          f"{'fc floor':>9} {'fc susc':>9} {'fc@edge axis':>13}")
    for r in rows:
        rel = np.asarray(r["lcc_frac_mean"])
        fi = fc_interpolated(r["fracs"], rel)
        fl = r["f_c_control_axis"]["lcc_half"]
        fe = r["f_c_edge_axis"]["lcc_half"]
        print(f"  {r['arm']:>22} {r['N']:>6} {r['k_nbr']:>4} "
              f"{(f'{fi:.4f}' if fi is not None else 'n/a'):>10} "
              f"{(f'{fl:.4f}' if fl is not None else 'n/a'):>9} "
              f"{r['f_c_control_axis']['susc_peak']:>9.4f} "
              f"{(f'{fe:.4f}' if fe is not None else 'n/a'):>13}")
    print("\n  'fc interp' is the published convention and is the column to "
          "compare\n  against 0.3210. 'fc floor' is sim04's own grid-floor "
          "value, biased up by\n  up to one grid spacing. For node arms, "
          "'fc@edge axis' is the one that is\n  comparable with the edge arms.")


def node_vs_edge():
    hdr("sim05 — node against edge attack (R1-3)")
    p = RUNS / "sim05_node_ablation" / "index.json"
    d = RUNS / "sim05_node_ablation"
    if not d.exists():
        print("  pending")
        return
    rows = []
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and (sub / "summary.json").exists():
            rows.append(json.loads((sub / "summary.json").read_text()))
    if not rows:
        print("  pending")
        return
    order = {p: i for i, p in enumerate(
        ("edge_targeted", "edge_random", "node_targeted_excise",
         "node_random_excise", "node_targeted_isolate", "node_random_isolate"))}
    rows.sort(key=lambda r: order.get(r["protocol"], 99))
    print(f"  {'protocol':>22} {'norm':>8} {'edges rm':>9} {'R_g s1':>8} "
          f"{'R_g s10':>9} {'drop %':>8} {'R_c s10':>9} {'chg %':>7}")
    for r in rows:
        print(f"  {r['protocol']:>22} {r['normalisation']:>8} "
              f"{r['edge_fraction_achieved_mean'][-1]*100:>8.1f}% "
              f"{r['R_global_mean'][0]:>8.4f} {r['R_global_mean'][-1]:>9.4f} "
              f"{r['R_global_drop_pct']:>8.1f} "
              f"{r['R_component_mean'][-1]:>9.4f} "
              f"{r['R_component_change_pct']:>7.2f}")
    exc = [r["R_global_drop_pct"] for r in rows if r["normalisation"] == "excise"]
    iso = [r["R_global_drop_pct"] for r in rows if r["normalisation"] == "isolate"]
    edg = [r["R_global_drop_pct"] for r in rows if r["normalisation"] == "n/a"]
    if exc and iso and edg:
        ratio = np.mean(iso) / np.mean(exc) if np.mean(exc) else float("nan")
        print(f"\n  NORMALISATION. Measured at comparable edge removal, the "
              f"excise protocols\n  report a mean R_g drop of {np.mean(exc):.1f}% "
              f"against {np.mean(iso):.1f}% for isolate, a factor of\n  "
              f"{ratio:.1f}. The damage is the same; the denominator is not. "
              "Excise deletes the\n  fragmented neurons and so removes them from "
              "the average.\n\n  Lead with 'isolate': it holds N fixed, so R_g "
              f"is the same quantity as in the\n  edge runs ({np.mean(edg):.1f}% "
              "mean drop) and the comparison is meaningful. Use\n  'excise' in the "
              "text to explain why R_g is not population-invariant.")


def main():
    validate()
    hdr("sim01 — network size (R1-2)")
    sweep_table("sim01_size_sweep", "N", "N")
    hdr("sim02 — average degree (R1-2)")
    sweep_table("sim02_degree_sweep", "k_nbr", "<k>")
    centrality()
    percolation()
    node_vs_edge()
    print()


if __name__ == "__main__":
    main()

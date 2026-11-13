# Revision campaigns

Code added during the revision of the manuscript, answering two referee questions that
the published scripts could not: how the results depend on network size and average
degree, and how node-based damage compares with edge-based damage.

Nothing in the parent directory was modified. `fhn_ext.py` imports the published model
and adds what these campaigns need on top of it, so the released code still reproduces
the originally submitted paper exactly.

## Layout

```
fhn_ext.py                  node ablation, a six-protocol ensemble driver, and a
                            cheap metrics path. Imports fhn_alzheimers from ../
sim01_size_sweep.py         N in {250, 500, 1000, 2000, 4000} at <k> = 8
sim02_degree_sweep.py       <k> in {4, 6, 8, 12} at N = 1000
sim03_centrality_stats.py   betweenness distribution against N and <k>, no dynamics
sim04_percolation_scaling.py  critical fraction against N and <k>, no dynamics
sim05_node_ablation.py      node-based against edge-based attack, six protocols
analyse.py                  reads runs/ and prints the tables, simulates nothing

figstyle.py                 fonts, palette, paths, and the shared panel-label helper
fhndata.py                  loads stored run output and derives plotted quantities
make_figures.py             Figures 1 to 10 of the paper
make_revision_figures.py    Figures 11 to 13 and Supplementary Figure S2
figures/                    the figures as they appear in the paper, PDF and PNG
runs/                       output of the campaigns above, read by analyse.py and
                            by the figure scripts
```

## Running

```bash
cd fhn-alzheimers/revision
python3 sim01_size_sweep.py          # or any other sim
python3 sim01_size_sweep.py --quick  # seconds, smoke test only, not results
python3 analyse.py                   # tables from whatever has finished
```

Every script takes `--networks`, `--trials`, `--stages`, `--outdir` and `--seed`. All
default to `--seed 20260529`, the master seed of the published runs, and to 10 networks
with 3 trials per stage, which is the published ensemble. Results are reproducible from
the seed alone.

Set `OMP_NUM_THREADS=1` when running several at once. The per-step work is a sparse
matrix-vector product on a vector of length N, too small to benefit from threading, so
threads only cause oversubscription.

Approximate wall times on 16 cores, running the five in parallel: sim03 and sim04 take
about 20 minutes each, sim05 about 2.3 hours, sim02 about 4 hours and sim01 about
5.8 hours, the last being dominated by the N = 4000 cell.

## Figures

```bash
python3 make_figures.py            # all available paper figures
python3 make_figures.py 2 10       # just those two
python3 make_revision_figures.py   # the revision figures
```

Both write PDF and PNG at 300 dpi into `figures/`. Every panel letter goes through
one helper in `figstyle.py`, so (a), (b), (c) sit in the same corner at the same size
on every figure.

Figures 1 to 4, 6, 10 to 13 and S2 build from a fresh clone, because the arrays they
need are committed under `runs/` and in the parent `runs/`. Figures 5 and 7 to 9 read
the full (v, w) traces, which are excluded by `.gitignore` because they run to about
10.7 GB. The script reports which it skipped and why rather than failing. To build
them, set `save_trajectories_at` in `Config` and rerun `fhn_alzheimers.py`.

The committed `figures/` directory holds all of them as they appear in the paper.

## Validation

`sim01`'s `N1000` cell and `sim05`'s `edge_targeted` cell both run at the published
canonical configuration and must reproduce the published numbers, R_g falling from
0.9685 to 0.5084 while R_c rises 0.35%. `analyse.py` checks this first and says so
loudly. If it fails, `fhn_ext.py` is wrong and nothing else here is trustworthy.

## Two design decisions worth knowing

**Node removal has two defensible normalisations, and both are run.** R_g is an average
over the population, so deleting neurons changes its denominator and node-attack R_g is
not directly comparable with edge-attack R_g. Under `excise` the node and its edges are
deleted and N shrinks, which is neuron death. Under `isolate` the node keeps its place
and loses every edge, so N is fixed and R_g stays the same quantity as in the edge runs.
A disconnected node is a component of size 1 under both and is excluded from R_c, which
is the published convention and is left alone.

The two differ by about a factor of four in the measured decline at the same edge
removal, so the choice is not cosmetic. The paper leads with `isolate` for that reason
and uses `excise` to explain why a global order parameter cannot be compared across
studies that prune their networks differently.

**Node fractions and edge fractions are different quantities.** Removing a fraction q of
nodes at random removes about 2q − q² of the edges, so the node arms run to a node
fraction of 0.60, which costs roughly the 85% of edges the published edge schedule
removes. Every stage records the edge fraction it actually achieved in
`edge_fraction_achieved`, and node and edge protocols should be compared against that
measured axis rather than against their own control parameter.

## Output

One directory per condition under the chosen `--outdir`:

| File | Contents |
|---|---|
| `data.npz` | `R_global` and `R_component`, each `(n_networks, n_stages, n_trials)`, plus `control_fraction` and `edge_fraction_achieved` |
| `metrics_per_network.json` | per-stage structural metrics for every network |
| `summary.json` | the full `Config`, stage means, SD across networks, endpoint percentage changes |

In `summary.json` the trial axis is averaged first and the network axis second, so
`R_global_sd_across_networks` is a spread across realisations.

`light_metrics` is used instead of the published `compute_graph_metrics`. The published
version also returns modularity, global efficiency and mean path length, each
superlinear in N, and those are what made the original canonical campaign take 150
minutes rather than the 17 its 300 simulations need. They are already reported at
N = 1000 in the paper, so the sweeps do not recompute them. `light_metrics` adds
`participation_ratio`, the PR of the decorrelated-phase reference level
R_g^pred = R_c sqrt(PR).

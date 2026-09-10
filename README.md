# Structural Fragmentation and the Dissociation of Global from Within-Component Synchrony

Simulation code and results for a stochastic FitzHugh–Nagumo small-world network
model of Alzheimer's disease.

The question the code answers: when a network's connections are destroyed but every
neuron is left untouched, what happens to synchrony? The answer is that the standard
global measure collapses while the within-component measure does not move, and the
gap between them separates structural damage from dynamical damage.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.13.9 with numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0, networkx 3.4.2.
Nothing else is needed. 

## Important: coupling normalization

`fhn_alzheimers.py` uses **degree-normalized** diffusive coupling:

```python
coupling = K * (A_csr @ v - deg * v) / deg_safe
```


## Layout

```
fhn_alzheimers.py         core model: network build, stochastic Heun integrator,
                          ablation protocol, order parameters. Everything imports this.
fhn_robustness.py         multi-network ensemble sweep -> the main results
fhn_percolation.py        fine-resolution edge-removal sweep -> percolation transition
fhn_diagnostics.py        phase distributions, mean-field spectra, autocorrelation
fhn_dt_convergence.py     timestep convergence check
fhn_window_convergence.py recording-window convergence check
fhn_hopf_criticality.py   single-neuron Hopf boundary and criticality sweeps
fhn_lyapunov.py           first Lyapunov coefficient at the Hopf point
replot_figures.py         rebuild the four npz-driven figures
regenerate_all_figures.py rebuild every paper figure (needs stored trajectories)
build_static_figures.py   rebuild the three figures that need no trajectories
analysis_scaling.py       Fokker-Planck ratio table, power-law fit, and the
                          component size-dependence check, all read from runs/

figures_normalized/       figures used in the paper
runs/                     saved output, one directory per campaign
```

The manuscript itself is not in this repository. This is the simulation code
and the run output behind it.

## Reproducing the results

Each command writes into its own output directory. Run them from the repository root.

`build_static_figures.py` and `analysis_scaling.py` work straight from a fresh
clone, because they read the committed arrays in `runs/`. `replot_figures.py` and `regenerate_all_figures.py`
read the raw output directories a run leaves behind, and `regenerate_all_figures.py`
additionally needs the stored trajectories, so both want the runs below done first.

| What | Command | Output |
|---|---|---|
| Canonical ensemble (n=10) | `python fhn_robustness.py --networks 10` | `alzheimers_robustness/` |
| Coupling sweep | `python fhn_robustness.py --K 0.20 --networks 5` (also 0.50) | per-K directory |
| Noise sweep | `python fhn_robustness.py --sigma 0.02 --networks 5` (also 0.10) | per-sigma directory |
| Sub-Hopf control | `python fhn_robustness.py --a 1.05 --networks 5` | `a105`-style directory |
| Random-ablation control | `python fhn_robustness.py --ablation random --networks 10` | random-ablation directory |
| Percolation sweep | `python fhn_percolation.py --n-fracs 30 --networks 2` | `alzheimers_percolation/` |
| Timestep convergence | `python fhn_dt_convergence.py` | `alzheimers_dt_convergence/` |
| Window convergence | `python fhn_window_convergence.py --T 6400 --networks 3 --trials 3` | `alzheimers_window_convergence_T6400/` |
| Hopf criticality | `python fhn_hopf_criticality.py` | `alzheimers_hopf/` |
| First Lyapunov coefficient | `python fhn_lyapunov.py` | prints `l1` and the verdict |
| Scaling and size-dependence analyses | `python analysis_scaling.py` | prints to stdout, reads `runs/` only |

Seeds are fixed (`master_seed = 20260529`), so a rerun reproduces the saved numbers.

## What is in `runs/`

| Directory | Campaign |
|---|---|
| `canonical/` | canonical parameters, n=10 networks, the headline result |
| `K020/`, `K050/` | coupling sweep, n=5 each |
| `s002/`, `s010/` | noise sweep, n=5 each |
| `a105/` | sub-Hopf control, no spontaneous oscillation |
| `random/` | random-ablation control |
| `percolation/` | fine-resolution sweep, 30 fractions |
| `canonical_figs/` | the run the paper's figures were drawn from |
| `convergence_normalized/` | timestep and window convergence, normalized coupling |
| `stationary/` | canonical ensemble with the window placed after the transient |
| `fp_followup/` | the (K, sigma) grid and the epsilon sweep behind the scaling analysis |
| `hopf/` | criticality sweeps for the single-neuron Hopf bifurcation |

Every directory keeps a `run.log` recording the exact configuration, and a
`*_summary.json` holding the numbers quoted in the paper.

## Please Note: 

Full `(v, w)` trajectories are excluded by `.gitignore`. There are 42 of them at
roughly 250 MB each, about 10.7 GB in total, which is far past GitHub's 100 MB
per-file limit. They are regenerable: set `save_trajectories_at` in `Config` and
rerun. Every number in the paper is derived from the summary files that *are*
committed, not from the raw traces.

The same applies to the 149 MB window-convergence trace. Its summary JSON alongside
it carries all the reported values.

## Reading the results

The two quantities to look for in any `*_summary.json`:

- `stats_R_global`, synchrony across all neurons, the conventional measure
- `stats_R_component`, synchrony within each connected piece, size-weighted

At canonical parameters the first falls by 47% across the ten damage stages while
the second rises slightly. That dissociation is the paper's central result.

## Licence

MIT. See `LICENSE`.

"""
Canonical ensemble with the recording window moved past the decorrelation
transient, so burn_in = 4000.
"""
import json, time, numpy as np
from pathlib import Path
from fhn_alzheimers import Config
from fhn_robustness import run_multi_network, stage_endpoint_tests

OUT = Path("out"); OUT.mkdir(exist_ok=True)
cfg = Config()
cfg.K, cfg.sigma, cfg.a = 0.35, 0.05, 0.65
cfg.burn_in = 4000.0        # skip the transient
cfg.T       = 800.0         # then average over a stationary window
cfg.n_trials = 3
cfg.raster_stages = (); cfg.save_trajectories_at = (); cfg.trajectory_outdir = ""
N_NET = 5
print(f"stationary canonical ensemble: burn_in={cfg.burn_in} T={cfg.T} "
      f"networks={N_NET} trials={cfg.n_trials}", flush=True)
t0 = time.time()
res = run_multi_network(cfg, N_NET)
print(f"wall-time {(time.time()-t0)/60:.1f} min", flush=True)
Rg, Rc = res["R_global"], res["R_component"]
out = dict(config=dict(K=cfg.K, sigma=cfg.sigma, a=cfg.a, burn_in=cfg.burn_in,
                       T=cfg.T, n_networks=N_NET, n_trials=cfg.n_trials),
           stats_R_global=stage_endpoint_tests(Rg, "R_global"),
           stats_R_component=stage_endpoint_tests(Rc, "R_component"))
json.dump(out, open(OUT/"stationary_summary.json","w"), indent=1, default=float)
np.savez_compressed(OUT/"stationary_data.npz", R_global=Rg, R_component=Rc)
g, c = out["stats_R_global"], out["stats_R_component"]
print(f"\nR_global    {g['stage1_mean']:.4f} -> {g['stageN_mean']:.4f}+-{g['stageN_std']:.4f}"
      f"  ({100*abs(g['delta_mean'])/g['stage1_mean']:.1f}% decline)")
print(f"R_component {c['stage1_mean']:.4f} -> {c['stageN_mean']:.4f}+-{c['stageN_std']:.4f}"
      f"  ({100*c['delta_mean']/c['stage1_mean']:+.2f}%)")

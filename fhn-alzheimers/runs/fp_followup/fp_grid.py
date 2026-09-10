"""
Log-spaced 5x5 grid in (K, sigma) at Stage 1, fitting 1-R = c K^p sigma^q
with p and q independent.
"""
import json, time, numpy as np
from pathlib import Path
from fhn_alzheimers import Config, build_initial_network, simulate

OUT = Path("grid_out"); OUT.mkdir(exist_ok=True)
N_NET, N_TRIAL = 3, 3
Ks     = np.geomspace(0.175, 0.70, 5)
Sigmas = np.geomspace(0.020, 0.10, 5)

base = Config()
root = np.random.SeedSequence(base.master_seed)
net_seeds = [int(s.generate_state(1)[0]) for s in root.spawn(N_NET)]
graphs = [build_initial_network(base, s) for s in net_seeds]
print(f"grid: {len(Ks)} K x {len(Sigmas)} sigma, {N_NET} networks x {N_TRIAL} trials, Stage 1 only")
print(f"K     = {[round(k,4) for k in Ks]}")
print(f"sigma = {[round(s,4) for s in Sigmas]}", flush=True)

rows = []; t0 = time.time()
for K in Ks:
    for sg in Sigmas:
        vals = []
        for ni, G in enumerate(graphs):
            cfg = Config(); cfg.K = float(K); cfg.sigma = float(sg)
            cfg.raster_stages = (); cfg.save_trajectories_at = (); cfg.trajectory_outdir = ""
            for k in range(N_TRIAL):
                seed = net_seeds[ni] * 10007 + int(K * 1e5) * 13 + int(sg * 1e5) * 7 + k
                vals.append(simulate(G, cfg, seed=seed)["R_component_mean"])
        v = np.array(vals)
        rows.append(dict(K=float(K), sigma=float(sg), R=float(v.mean()),
                         R_sd=float(v.std(ddof=1)), n=len(v)))
        print(f"  K={K:.4f} sigma={sg:.4f}  R={v.mean():.5f}+-{v.std(ddof=1):.5f}  "
              f"1-R={1-v.mean():.5f}  ({time.time()-t0:.0f}s)", flush=True)

K_ = np.array([r["K"] for r in rows]); S_ = np.array([r["sigma"] for r in rows])
y  = 1 - np.array([r["R"] for r in rows])
A  = np.column_stack([np.ones(len(y)), np.log(K_), np.log(S_)])
coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
c, p, q = coef
resid = np.log(y) - A @ coef
dof = len(y) - 3
cov = (resid @ resid / dof) * np.linalg.inv(A.T @ A)
sp, sq = np.sqrt(cov[1, 1]), np.sqrt(cov[2, 2])
R2 = 1 - (resid**2).sum() / ((np.log(y) - np.log(y).mean())**2).sum()
# q + 2p should be 0 under the hypothesis
var_comb = cov[2,2] + 4*cov[1,1] + 4*cov[1,2]
sc = np.sqrt(max(var_comb, 0.0))
print("\n" + "="*66)
print(f"INDEPENDENT FIT   1-R = {np.exp(c):.4f} * K^p * sigma^q     (n={len(y)})")
print(f"   p = {p:+.4f} +- {sp:.4f}")
print(f"   q = {q:+.4f} +- {sq:.4f}")
print(f"   R^2(log) = {R2:.5f}")
print(f"   q + 2p = {q+2*p:+.4f} +- {sc:.4f}   (sigma^2/K requires 0)  -> {abs(q+2*p)/sc if sc>0 else float('nan'):.2f} sigma")
print(f"   effective sigma^2/K exponent = {-p:.4f} (from p),  {q/2:.4f} (from q)")
json.dump(dict(rows=rows, fit=dict(c=float(np.exp(c)), p=float(p), q=float(q),
               p_se=float(sp), q_se=float(sq), R2=float(R2),
               q_plus_2p=float(q+2*p), q_plus_2p_se=float(sc))),
          open(OUT/"grid_summary.json","w"), indent=1)
np.savez_compressed(OUT/"grid_data.npz", K=K_, sigma=S_, R=1-y)
print(f"\nTotal wall-time: {(time.time()-t0)/60:.1f} min")

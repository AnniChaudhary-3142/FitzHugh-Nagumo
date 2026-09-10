"""
Repeats the five-condition Fokker-Planck campaign at eps = 0.06, 0.30 and
0.80, testing whether the exponent discrepancy sits in the harmonic
truncation.
"""
import json, time, numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from fhn_alzheimers import Config, build_initial_network, simulate

OUT = Path("eps_out_v2"); OUT.mkdir(exist_ok=True)
N_NET, N_TRIAL = 3, 3
B, I = 0.80, 0.30
CONDS = [(0.20,0.05),(0.35,0.05),(0.50,0.05),(0.35,0.02),(0.35,0.10)]

def a_hopf(eps):
    v = -np.sqrt(1 - eps*B)
    return B*(v - v**3/3 + I) - v

def period(a, eps, T=6000):
    f = lambda t,y: [y[0]-y[0]**3/3-y[1]+I, eps*(y[0]+a-B*y[1])]
    s = solve_ivp(f,[0,T],[0.1,0.0],max_step=0.05,rtol=1e-10,atol=1e-12,dense_output=True)
    t = np.linspace(T*0.6, T, 400000); v = s.sol(t)[0]
    vc = v - v.mean(); sign = np.sign(vc)
    up = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    return float(np.mean(np.diff(t[up]))) if len(up) > 2 else np.nan

DIST = a_hopf(0.06) - 0.65          # matched distance below Hopf
P0   = period(0.65, 0.06)
print(f"canonical: a_Hopf(0.06)={a_hopf(0.06):.4f}, a=0.65, distance={DIST:.4f}, period={P0:.2f}", flush=True)

results = {}
t0 = time.time()
for eps in (0.06, 0.30, 0.80):
    a = a_hopf(eps) - DIST
    P = period(a, eps)
    # do not scale windows with the period: at eps=0.30, burn_in 32 gives
    # SD 0.0196 against 0.0001 at burn_in 300
    scale = P / P0
    T_eps  = 1200.0
    bi_eps = 300.0
    print(f"\n=== eps={eps}  a_Hopf={a_hopf(eps):.4f}  a={a:.4f}  period={P:.2f}  "
          f"scale={scale:.3f}  T={T_eps:.1f} burn_in={bi_eps:.1f} ===", flush=True)
    # --- calibrate K so the operating point (R at sigma=0.05) matches canonical ---
    def R_at(Kv):
        c = Config(); c.a=a; c.eps=eps; c.K=float(Kv); c.sigma=0.05
        c.T=T_eps; c.burn_in=bi_eps; c.K_ramp_time=20.0
        c.raster_stages=(); c.save_trajectories_at=(); c.trajectory_outdir=""
        Gc = build_initial_network(c, 12345)
        return simulate(Gc, c, seed=7)["R_component_mean"]
    TARGET = 0.9685
    lo, hi = 0.35, 0.35
    while R_at(hi) < TARGET and hi < 40: hi *= 1.6
    while R_at(lo) > TARGET and lo > 0.02: lo /= 1.6
    for _ in range(12):
        mid = np.sqrt(lo*hi)
        if R_at(mid) < TARGET: lo = mid
        else: hi = mid
    K_ref = float(np.sqrt(lo*hi)); fK = K_ref/0.35
    print(f"   calibrated K_ref={K_ref:.4f} (factor {fK:.3f}) so R(sigma=0.05)~{TARGET}", flush=True)
    base = Config(); base.a = a; base.eps = eps
    root = np.random.SeedSequence(base.master_seed)
    net_seeds = [int(s.generate_state(1)[0]) for s in root.spawn(N_NET)]
    graphs = [build_initial_network(base, s) for s in net_seeds]
    rows = []
    for K0, sg in CONDS:
        K = K0*fK
        vals = []
        for ni, G in enumerate(graphs):
            cfg = Config(); cfg.a=a; cfg.eps=eps; cfg.K=K; cfg.sigma=sg
            cfg.T=T_eps; cfg.burn_in=bi_eps; cfg.K_ramp_time=20.0
            cfg.raster_stages=(); cfg.save_trajectories_at=(); cfg.trajectory_outdir=""
            for k in range(N_TRIAL):
                seed = net_seeds[ni]*10007 + int(K*1e5)*13 + int(sg*1e5)*7 + k
                vals.append(simulate(G, cfg, seed=seed)["R_component_mean"])
        v = np.array(vals); R = v.mean()
        rows.append(dict(K=float(K), K0=K0, sigma=sg, R=float(R),
                         R_sd=float(v.std(ddof=1)), ratio=float((1-R)*K/sg**2)))
        print(f"   K={K:.2f} sigma={sg:.3f}  R={R:.5f}+-{v.std(ddof=1):.5f}  "
              f"(1-R)K/s^2={(1-R)*K/sg**2:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    x = np.array([r["sigma"]**2/r["K"] for r in rows])
    y = np.array([1-r["R"] for r in rows])
    expo, logc = np.polyfit(np.log(x), np.log(y), 1)
    resid = np.log(y) - np.polyval([expo,logc], np.log(x))
    R2 = 1 - (resid**2).sum()/((np.log(y)-np.log(y).mean())**2).sum()
    ratios = np.array([r["ratio"] for r in rows])
    spread = 100*ratios.std(ddof=1)/ratios.mean()
    results[str(eps)] = dict(a=a, a_hopf=a_hopf(eps), period=P, T=T_eps, K_ref=K_ref, fK=fK,
                             rows=rows, exponent=float(expo), c=float(np.exp(logc)),
                             R2=float(R2), ratio_mean=float(ratios.mean()),
                             ratio_sd=float(ratios.std(ddof=1)), spread_pct=float(spread))
    print(f"   -> exponent={expo:.4f}  R2={R2:.5f}  ratio={ratios.mean():.3f}+-{ratios.std(ddof=1):.3f} (+-{spread:.1f}%)", flush=True)

print("\n" + "="*66)
print(f"{'eps':>6} {'a':>8} {'exponent':>10} {'R2':>8} {'spread':>9}")
for e,r in results.items():
    print(f"{e:>6} {r['a']:8.4f} {r['exponent']:10.4f} {r['R2']:8.5f} {r['spread_pct']:8.1f}%")
print("\nPREDICTION if localization is correct: exponent -> 1, spread shrinks monotonically with eps")
json.dump(results, open(OUT/"eps_sweep_summary.json","w"), indent=1)
print(f"Total wall-time: {(time.time()-t0)/60:.1f} min")

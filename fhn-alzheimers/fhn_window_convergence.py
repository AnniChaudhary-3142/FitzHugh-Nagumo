"""
Checks whether the Stage 10 R_global endpoint is settled at T = 400 or
still decaying, by comparing prefix means against block means on long runs.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from fhn_alzheimers import Config, build_initial_network, simulate


T_CANONICAL = 400.0     # the window the paper actually reports


# ---- sweep ----
def run_window_sweep(T_long, block, stages, n_networks, n_trials, seed,
                     eps_spread=0.0):
    cfg_base = Config()
    cfg_base.master_seed = seed

    # same seeding as fhn_robustness, so network 0 is the canonical graph
    root = np.random.SeedSequence(cfg_base.master_seed)
    net_seedseqs = root.spawn(n_networks)

    n_rec = int(round(T_long / cfg_base.dt))
    traces_g = {s: np.empty((n_networks, n_trials, n_rec)) for s in stages}
    traces_c = {s: np.empty((n_networks, n_trials, n_rec)) for s in stages}
    pr_bound = {s: np.zeros(n_networks) for s in stages}
    comp_info = {s: [] for s in stages}

    for ni, net_ss in enumerate(net_seedseqs):
        net_seed = int(net_ss.generate_state(1)[0])
        G0 = build_initial_network(cfg_base, net_seed)
        print(f"\n{'='*64}")
        print(f"NETWORK {ni+1}/{n_networks}  (seed={net_seed})")
        print(f"{'='*64}")
        t0 = time.time()
        eb = nx.edge_betweenness_centrality(G0)
        edge_order = sorted(eb, key=eb.get, reverse=True)
        print(f"  betweenness done in {time.time()-t0:.1f}s, |E|_0={len(edge_order)}")

        for stage in stages:
            cfg = Config()
            cfg.master_seed = seed
            cfg.T = T_long
            cfg.raster_stages = ()
            cfg.save_trajectories_at = ()
            cfg.trajectory_outdir = ""
            # detuning belongs to the neurons, so it is keyed to the network
            cfg.eps_spread = eps_spread
            cfg.eps_seed = net_seed

            frac = cfg.ablation_fraction(stage)
            G = G0.copy()
            G.remove_edges_from(edge_order[:int(round(frac * len(edge_order)))])

            sizes = np.array([len(c) for c in nx.connected_components(G)])
            PR = float((sizes ** 2).sum()) / cfg.N ** 2
            pr_bound[stage][ni] = PR
            comp_info[stage].append(dict(n_components=int(len(sizes)),
                                          lcc=int(sizes.max()),
                                          PR=PR,
                                          edges=int(G.number_of_edges())))

            print(f"\n  --- Stage {stage}  f={frac:.2f}  |E|={G.number_of_edges()}"
                  f"  comps={len(sizes)}  LCC={sizes.max()}  PR={PR:.5f} ---")

            for k in range(n_trials):
                # offset by 50000 to stay disjoint from the dt-convergence seeds
                trial_seed = net_seed * 10000 + 50000 + stage * 100 + k
                t0 = time.time()
                res = simulate(G, cfg, seed=trial_seed, store_full=False)
                traces_g[stage][ni, k] = res["R_global_trace"]
                traces_c[stage][ni, k] = res["R_component_trace"]
                n_canon = int(round(T_CANONICAL / cfg.dt))
                pre = res["R_global_trace"][:n_canon].mean()
                full = res["R_global_trace"].mean()
                print(f"    trial {k}: R_g[0:{T_CANONICAL:.0f}]={pre:.4f}  "
                      f"R_g[0:{T_long:.0f}]={full:.4f}  ({time.time()-t0:.1f}s)")

    return traces_g, traces_c, pr_bound, comp_info


# ---- analysis ----
def analyse(trace_3d, dt, block):
    """Flatten (n_networks, n_trials, n_rec) into observations. Returns prefix
    means, block means, and a per-observation slope across blocks."""
    n_net, n_tr, n_rec = trace_3d.shape
    flat = trace_3d.reshape(n_net * n_tr, n_rec)
    n_per = int(round(block / dt))
    n_blocks = n_rec // n_per

    prefix = np.array([[flat[k, : (w + 1) * n_per].mean()
                        for w in range(n_blocks)] for k in range(flat.shape[0])])
    blocks = np.array([[flat[k, w * n_per : (w + 1) * n_per].mean()
                        for w in range(n_blocks)] for k in range(flat.shape[0])])

    x = np.arange(n_blocks)
    slopes = np.array([np.polyfit(x, blocks[k], 1)[0]
                       for k in range(flat.shape[0])])
    return prefix, blocks, slopes


def fit_asymptote(block_mean, block, T_long):
    """Fit R(t) = R_inf + A*exp(-t/tau) to the block means. None if it fails."""
    n_blocks = len(block_mean)
    if n_blocks < 5:
        return None
    t = (np.arange(n_blocks) + 0.5) * block

    def model(tt, R_inf, A, tau):
        return R_inf + A * np.exp(-tt / tau)

    try:
        p0 = [float(np.clip(block_mean[-1], 0.0, 1.0)),
              float(np.clip(block_mean[0] - block_mean[-1], 1e-6, 1.0)),
              T_long / 3.0]
        popt, pcov = curve_fit(model, t, block_mean, p0=p0, maxfev=20000,
                               bounds=([0.0, 0.0, 1.0], [1.0, 1.0, 1e6]))
    except Exception as exc:  # fit is diagnostic only, never fatal
        return dict(error=str(exc))

    perr = np.sqrt(np.diag(pcov))
    resid = block_mean - model(t, *popt)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((block_mean - block_mean.mean()) ** 2).sum())
    return dict(
        R_inf=float(popt[0]), R_inf_se=float(perr[0]),
        A=float(popt[1]), tau=float(popt[2]), tau_se=float(perr[2]),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        # tau well inside the window means the decay is actually resolved
        tau_resolved=bool(popt[2] < 0.5 * T_long),
    )


# ---- plot ----
def plot_windows(traces_g, dt, block, stages, T_long, fits, out_path):
    fig, axes = plt.subplots(len(stages), 2,
                             figsize=(13.5, 4.6 * len(stages)), squeeze=False)

    for si, stage in enumerate(stages):
        tr = traces_g[stage]
        n_rec = tr.shape[2]
        t = np.arange(n_rec) * dt
        prefix, blocks, slopes = analyse(tr, dt, block)
        n_blocks = blocks.shape[1]

        ax = axes[si][0]
        flat = tr.reshape(-1, n_rec)
        run = np.cumsum(flat, axis=1) / np.arange(1, n_rec + 1)
        mu, sd = run.mean(axis=0), run.std(axis=0)
        ax.plot(t, mu, color="#1f77b4", lw=2.0, label="running mean")
        ax.fill_between(t, mu - sd, mu + sd, color="#1f77b4", alpha=0.22,
                        label="±1 SD across runs")
        ax.axvline(T_CANONICAL, color="crimson", ls="--", lw=1.5,
                   label=f"canonical window (T={T_CANONICAL:.0f})")
        ax.set_xlabel("Recording time")
        ax.set_ylabel("Running mean of $R_{global}$")
        ax.set_title(f"Stage {stage}: prefix-mean convergence")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        ax = axes[si][1]
        tb = (np.arange(n_blocks) + 0.5) * block
        ax.errorbar(tb, blocks.mean(axis=0), yerr=blocks.std(axis=0),
                    fmt="o-", color="#2ca02c", lw=2.0, ms=6, capsize=4,
                    label="block mean")
        ax.errorbar(tb, prefix.mean(axis=0), yerr=prefix.std(axis=0),
                    fmt="s--", color="#7f7f7f", lw=1.4, ms=5, capsize=3,
                    alpha=0.8, label="prefix mean")

        ft = fits.get(stage)
        if ft and "R_inf" in ft:
            tt = np.linspace(0, T_long, 400)
            ax.plot(tt, ft["R_inf"] + ft["A"] * np.exp(-tt / ft["tau"]),
                    color="#d62728", lw=1.6, ls=":",
                    label=(f"fit: $R_\\infty$={ft['R_inf']:.3f}, "
                           f"$\\tau$={ft['tau']:.0f}"))
            ax.axhline(ft["R_inf"], color="#d62728", lw=1.0, alpha=0.5)
        if stage in fits and fits[stage] and "pr_bound" in fits[stage]:
            ax.axhline(fits[stage]["pr_bound"], color="#9467bd", lw=1.4,
                       ls="-.", label=f"PR bound = {fits[stage]['pr_bound']:.3f}")

        ax.set_xlabel(f"Recording time (blocks of {block:.0f})")
        ax.set_ylabel("$R_{global}$")
        ax.set_title(f"Stage {stage}: slope = {slopes.mean():+.5f} "
                     f"± {slopes.std():.5f} per {block:.0f} t.u.")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Recording-window convergence: canonical T={T_CANONICAL:.0f} "
                 f"against T={T_long:.0f}", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1600.0,
                    help="Long recording window (canonical is 400)")
    ap.add_argument("--block", type=float, default=400.0,
                    help="Analysis block width")
    ap.add_argument("--networks", type=int, default=1)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--stages", type=int, nargs="+", default=[10, 1],
                    help="Stages to run; 10 is the case in question, 1 the reference")
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--eps-spread", type=float, default=0.0,
                    help="Relative SD of per-neuron eps heterogeneity (0 = homogeneous)")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    stages = tuple(args.stages)
    _tag = f"_eps{args.eps_spread:g}" if args.eps_spread > 0 else ""
    out_dir = Path(args.outdir) if args.outdir else Path(
        f"alzheimers_window_convergence_T{int(args.T)}{_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = Config().dt

    print("\nRECORDING-WINDOW CONVERGENCE CHECK")
    print(f"  T_canonical={T_CANONICAL}  T_long={args.T}  block={args.block}")
    print(f"  stages={stages}  networks={args.networks}  trials={args.trials}")
    print(f"  eps_spread={args.eps_spread}")
    print(f"  outdir={out_dir}")

    t_start = time.time()
    traces_g, traces_c, pr, comp_info = run_window_sweep(
        args.T, args.block, stages, args.networks, args.trials, args.seed,
        eps_spread=args.eps_spread)
    print(f"\nTotal wall-time: {(time.time()-t_start)/60.0:.1f} min")

    np.savez_compressed(
        out_dir / "window_convergence_data.npz",
        dt=dt, T_long=args.T, T_canonical=T_CANONICAL, block=args.block,
        stages=np.array(stages),
        **{f"R_global_stage{s}": traces_g[s] for s in stages},
        **{f"R_component_stage{s}": traces_c[s] for s in stages},
        **{f"PR_stage{s}": pr[s] for s in stages},
    )

    summary = dict(T_canonical=T_CANONICAL, T_long=args.T, block=args.block,
                   n_networks=args.networks, n_trials=args.trials,
                   eps_spread=args.eps_spread,
                   components={str(s): comp_info[s] for s in stages},
                   stages={})
    fits_for_plot = {}

    for stage in stages:
        entry = {}
        Rc_mean = float(traces_c[stage].mean())
        pr_b = float(np.mean(Rc_mean * np.sqrt(pr[stage])))

        for name, tr in (("R_global", traces_g[stage]),
                         ("R_component", traces_c[stage])):
            prefix, blocks, slopes = analyse(tr, dt, args.block)
            n_blocks = blocks.shape[1]
            bm = blocks.mean(axis=0)

            n_canon_blocks = min(max(int(round(T_CANONICAL / args.block)), 1),
                                 n_blocks)
            p_canon = prefix[:, n_canon_blocks - 1]
            p_full = prefix[:, -1]
            delta = p_full - p_canon
            b_delta = blocks[:, -1] - blocks[:, 0]

            ft = fit_asymptote(bm, args.block, args.T) if name == "R_global" else None

            entry[name] = dict(
                prefix_mean=[float(v) for v in prefix.mean(axis=0)],
                block_mean=[float(v) for v in bm],
                block_sd=[float(v) for v in blocks.std(axis=0)],
                canonical_window_mean=float(p_canon.mean()),
                full_window_mean=float(p_full.mean()),
                prefix_shift=float(delta.mean()), prefix_shift_sd=float(delta.std()),
                block_slope=float(slopes.mean()), block_slope_sd=float(slopes.std()),
                first_last_block_delta=float(b_delta.mean()),
                first_last_block_delta_sd=float(b_delta.std()),
                n_blocks=int(n_blocks),
            )
            if ft:
                entry[name]["asymptote_fit"] = ft

            print(f"\n--- Stage {stage} / {name} ---")
            print("  block |   t range    | prefix mean | block mean")
            for w in range(n_blocks):
                print(f"   {w+1:>3}  | {w*args.block:>5.0f}-{(w+1)*args.block:>5.0f} "
                      f"|   {prefix.mean(axis=0)[w]:.4f}    |   {bm[w]:.4f}")
            print(f"  prefix shift [0,{T_CANONICAL:.0f}] -> [0,{args.T:.0f}] = "
                  f"{delta.mean():+.4f} ± {delta.std():.4f}")
            print(f"  block slope = {slopes.mean():+.5f} ± {slopes.std():.5f} "
                  f"per {args.block:.0f} t.u.")
            print(f"  block 1 -> block {n_blocks} = "
                  f"{b_delta.mean():+.4f} ± {b_delta.std():.4f}")

            sd = b_delta.std()
            stable = abs(b_delta.mean()) <= max(2.0 * sd, 0.005)
            entry[name]["window_stable"] = bool(stable)
            print(f"  verdict: {'WINDOW-STABLE' if stable else 'STILL DRIFTING'}")

            if ft and "R_inf" in ft:
                print(f"  asymptote fit: R_inf={ft['R_inf']:.4f} "
                      f"± {ft['R_inf_se']:.4f}   tau={ft['tau']:.0f} "
                      f"± {ft['tau_se']:.0f}   R^2={ft['r_squared']:.4f}")
                print(f"    tau resolved within window: {ft['tau_resolved']}")
                print(f"    PR bound R_c*sqrt(PR) = {pr_b:.4f}   "
                      f"(R_inf - bound = {ft['R_inf']-pr_b:+.4f})")
            elif ft:
                print(f"  asymptote fit failed: {ft.get('error')}")

        entry["pr_bound"] = pr_b
        entry["R_component_mean"] = Rc_mean
        summary["stages"][str(stage)] = entry
        fits_for_plot[stage] = dict(entry["R_global"].get("asymptote_fit") or {},
                                     pr_bound=pr_b)

    plot_windows(traces_g, dt, args.block, stages, args.T,
                 fits_for_plot, out_dir / "window_convergence.png")

    with open(out_dir / "window_convergence_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n✓ All outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
Phase-difference distributions, mean-field spectra and periodicity
measures, computed from stored trajectories.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.signal import welch


# ---- must match the canonical run that produced the trajectories ----
TRAJ_DIR = Path("alzheimers_results/trajectories_canonical")
OUT_DIR      = Path("alzheimers_diagnostics")
STAGES       = [1, 3, 5, 7, 10]

N            = 1000
K_NBR        = 8
P_REWIRE     = 0.1
N_STAGES     = 10
ABLATION_MAX = 0.85
MASTER_SEED  = 20260529
A_OSC        = 0.65
B            = 0.80
I_EXT        = 0.30


# ---- helpers shared with fhn_alzheimers.py ----
def fhn_fixed_point(a, b, I):
    def f(v): return v - v ** 3 / 3.0 + I - (v + a) / b
    grid = np.linspace(-3.0, 3.0, 601)
    sign_changes = np.where(np.diff(np.sign(f(grid))))[0]
    idx = sign_changes[0]
    v_star = brentq(f, grid[idx], grid[idx + 1])
    w_star = (v_star + a) / b
    return v_star, w_star


def reconstruct_graph_at_stage(stage: int) -> nx.Graph:
    """Rebuild the ablated graph at one stage, matching fhn_alzheimers.py."""
    root = np.random.SeedSequence(MASTER_SEED)
    net_ss = root.spawn(2)[0]
    g0_seed = int(net_ss.generate_state(1)[0])
    G0 = nx.watts_strogatz_graph(N, k=K_NBR, p=P_REWIRE, seed=g0_seed)

    eb = nx.edge_betweenness_centrality(G0)
    edge_order = sorted(eb, key=eb.get, reverse=True)

    alpha = (stage - 1) / (N_STAGES - 1) if N_STAGES > 1 else 0.0
    frac = alpha * ABLATION_MAX
    n_remove = int(round(frac * len(edge_order)))

    G = G0.copy()
    G.remove_edges_from(edge_order[:n_remove])
    return G


def load_trajectory(stage: int) -> dict:
    path = TRAJ_DIR / f"traj_stage{stage:02d}_trial0.npz"
    data = np.load(path)
    return dict(
        V=data["V"],
        W=data["W"],
        dt=float(data["dt"]),
        stage=int(data["stage"]),
        K=float(data["K"]),
        sigma=float(data["sigma"]),
        a=float(data["a"]),
    )


# ---- diagnostic 1: phase-difference distributions ----
def plot_phase_distributions(stages_data, out_path):
    """Within-component phase deviations on top, between-component mean-phase
    differences below. The bottom row is centred, since a stage with one
    component has no pairs to show."""
    v_star, w_star = fhn_fixed_point(A_OSC, B, I_EXT)
    plot_stages = [s for s in [1, 5, 10] if s in stages_data]
    n = len(plot_stages)

    # ---- pass 1: compute, so we know which stages have pairwise data ----
    within_by_stage = {}
    pairwise_by_stage = {}
    for stage in plot_stages:
        d = stages_data[stage]
        V, W = d["V"], d["W"]

        # Time-average over the last quarter of the trace (steady-state regime)
        t_start = int(0.75 * V.shape[0])
        snapshots = V.shape[0] - t_start
        sample_idxs = np.linspace(t_start, V.shape[0] - 1, min(50, snapshots)).astype(int)

        # Components from reconstructed graph
        G = reconstruct_graph_at_stage(stage)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        big_components = [np.array(sorted(c), dtype=np.int64)
                          for c in components if len(c) >= 2]

        within_dev_all = []
        pairwise_diff_all = []
        pairwise_weight_all = []

        for ti in sample_idxs:
            phases = np.arctan2(W[ti] - w_star, V[ti] - v_star)
            mean_phases = []
            sizes = []
            for comp in big_components:
                cp = phases[comp]
                z = np.mean(np.exp(1j * cp))
                mph = np.angle(z)
                mean_phases.append(mph)
                sizes.append(len(comp))
                dev = np.angle(np.exp(1j * (cp - mph)))
                within_dev_all.extend(dev)

            if len(big_components) >= 2:
                mean_phases = np.array(mean_phases)
                sizes = np.array(sizes)
                for ii in range(len(mean_phases)):
                    for kk in range(ii + 1, len(mean_phases)):
                        diff = np.angle(np.exp(1j * (mean_phases[ii] - mean_phases[kk])))
                        pairwise_diff_all.append(diff)
                        pairwise_weight_all.append(sizes[ii] * sizes[kk])

        within_by_stage[stage] = within_dev_all
        if len(pairwise_diff_all) > 0:
            pairwise_by_stage[stage] = (pairwise_diff_all, pairwise_weight_all)

    # each panel spans 2 half-columns; the bottom row is offset by (n - m) to centre
    bottom_stages = [s for s in plot_stages if s in pairwise_by_stage]
    m = len(bottom_stages)
    offset = n - m

    fig = plt.figure(figsize=(5 * n, 8))
    gs = fig.add_gridspec(2, 2 * n)

    for j, stage in enumerate(plot_stages):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        ax.hist(within_by_stage[stage], bins=60, range=(-np.pi, np.pi),
                color="C0", alpha=0.85, density=True)
        ax.set_xlabel("Phase deviation from component mean (rad)")
        ax.set_ylabel("density")
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)

    for j, stage in enumerate(bottom_stages):
        c0 = offset + 2 * j
        ax = fig.add_subplot(gs[1, c0:c0 + 2])
        diffs, weights = pairwise_by_stage[stage]
        ax.hist(diffs, bins=60, range=(-np.pi, np.pi),
                weights=weights, color="C3", alpha=0.85, density=True)
        ax.axhline(1.0 / (2 * np.pi), color="k", linestyle="--", lw=1,
                   label="uniform baseline 1/2\u03c0")
        ax.set_xlabel("Pairwise mean-phase difference  (size\u00b2 weighted)")
        ax.set_ylabel("density")
        ax.set_xlim(-np.pi, np.pi)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- diagnostic 2: mean-field power spectra ----
def plot_mean_field_spectra(stages_data, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    cmap = plt.cm.viridis
    sorted_stages = sorted(stages_data.keys())

    peak_summary = {}
    for i, stage in enumerate(sorted_stages):
        d = stages_data[stage]
        mean_v = d["V"].mean(axis=1)
        dt = d["dt"]
        nperseg = min(8192, len(mean_v) // 4)
        freqs, psd = welch(mean_v, fs=1.0 / dt, nperseg=nperseg)

        color = cmap(i / max(len(sorted_stages) - 1, 1))
        ax.semilogy(freqs, psd, color=color, lw=1.7, label=f"Stage {stage}")

        band = (freqs > 0.01) & (freqs < 0.5)
        peak_idx = np.argmax(psd[band])
        peak_summary[stage] = dict(
            f_peak=float(freqs[band][peak_idx]),
            psd_peak=float(psd[band][peak_idx]),
            psd_total=float(np.trapezoid(psd, freqs)),
        )

    ax.set_xlim(0, 0.4)
    ax.set_xlabel("Frequency  (1/time unit)")
    ax.set_ylabel("Power spectral density of  ⟨v⟩(t)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return peak_summary


# ---- diagnostic 3: variance and autocorrelation ----
def plot_periodicity_diagnostics(stages_data, out_path):
    sorted_stages = sorted(stages_data.keys())
    variances = []
    autocorr_traces = {}

    for stage in sorted_stages:
        d = stages_data[stage]
        mean_v = d["V"].mean(axis=1)
        dt = d["dt"]
        variances.append(float(np.var(mean_v)))

        mv = mean_v - mean_v.mean()
        nfft = 2 ** int(np.ceil(np.log2(2 * len(mv))))
        f = np.fft.fft(mv, n=nfft)
        ac = np.fft.ifft(f * np.conj(f)).real[:len(mv)]
        ac = ac / ac[0]
        lag_steps = min(len(ac), int(60.0 / dt))      # show ~60 time units of lag
        autocorr_traces[stage] = (np.arange(lag_steps) * dt, ac[:lag_steps])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    ax.plot(sorted_stages, variances, "o-", lw=2.2, ms=9, color="C0")
    ax.set_xlabel("Disease stage")
    ax.set_ylabel("Var(⟨v⟩(t))")
    ax.set_xticks(sorted_stages)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cmap = plt.cm.viridis
    for i, stage in enumerate(sorted_stages):
        lags, ac = autocorr_traces[stage]
        color = cmap(i / max(len(sorted_stages) - 1, 1))
        ax.plot(lags, ac, color=color, lw=1.6, label=f"Stage {stage}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Lag  τ")
    ax.set_ylabel("Normalised autocorrelation  C(τ)/C(0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return variances


# ---- main ----
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectories from {TRAJ_DIR}...")
    stages_data = {}
    for stage in STAGES:
        path = TRAJ_DIR / f"traj_stage{stage:02d}_trial0.npz"
        if not path.exists():
            print(f"  [WARN] missing: {path}")
            continue
        stages_data[stage] = load_trajectory(stage)
        d = stages_data[stage]
        print(f"  Stage {stage}: V.shape={d['V'].shape}  "
              f"K={d['K']:.3f}  σ={d['sigma']:.3f}  a={d['a']:.3f}")

    if not stages_data:
        raise SystemExit("No trajectory files found. Check TRAJ_DIR.")

    print("\nDiagnostic 1 — phase-difference distributions ...")
    plot_phase_distributions(stages_data, OUT_DIR / "phase_distributions.png")

    print("Diagnostic 2 — mean-field power spectra ...")
    peak = plot_mean_field_spectra(stages_data, OUT_DIR / "mean_field_spectra.png")
    print("  Peak summary:")
    for s, info in peak.items():
        print(f"    Stage {s:2d}:  f_peak = {info['f_peak']:.4f}   "
              f"PSD_peak = {info['psd_peak']:.3e}   "
              f"total PSD = {info['psd_total']:.3e}")

    print("Diagnostic 3 — variance & autocorrelation ...")
    variances = plot_periodicity_diagnostics(
        stages_data, OUT_DIR / "periodicity_diagnostics.png"
    )
    print("  Var(⟨v⟩) by stage:")
    for s, v in zip(sorted(stages_data), variances):
        print(f"    Stage {s:2d}:  Var = {v:.4e}")

    print(f"\n✓ All diagnostic figures in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

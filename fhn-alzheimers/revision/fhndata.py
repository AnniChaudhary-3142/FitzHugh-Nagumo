"""
Data access and derived quantities for the revision figures.

Everything is read from stored run output. No simulation is re-run except
the two-variable isolated-neuron integration behind Figure 1, which is
deterministic given the seed used in the published code.

Source directories are listed in figstyle.py. Heavy derived quantities are
cached under revision/cache/ so make_figures.py is cheap to re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import brentq
from scipy.signal import welch

import figstyle as S

CACHE = S.HERE / "cache"


# --------------------------------------------------------------------------
# model helpers, copied from fhn_alzheimers.py / fhn_diagnostics.py
# --------------------------------------------------------------------------
def fhn_fixed_point(a=S.A_OSC, b=S.B, I=S.I_EXT) -> tuple[float, float]:
    def f(v):
        return v - v ** 3 / 3.0 + I - (v + a) / b
    grid = np.linspace(-3.0, 3.0, 601)
    idx = np.where(np.diff(np.sign(f(grid))))[0][0]
    v_star = brentq(f, grid[idx], grid[idx + 1])
    return v_star, (v_star + a) / b


def ablation_fraction(stage: int) -> float:
    alpha = (stage - 1) / (S.N_STAGES - 1)
    return alpha * S.ABLATION_MAX


def reconstruct_graph_at_stage(stage: int) -> nx.Graph:
    """Rebuild the ablated graph at one stage, matching fhn_alzheimers.py."""
    root = np.random.SeedSequence(S.MASTER_SEED)
    net_ss = root.spawn(2)[0]
    g0_seed = int(net_ss.generate_state(1)[0])
    G0 = nx.watts_strogatz_graph(S.N, k=S.K_NBR, p=S.P_REWIRE, seed=g0_seed)

    eb = nx.edge_betweenness_centrality(G0)
    edge_order = sorted(eb, key=eb.get, reverse=True)

    n_remove = int(round(ablation_fraction(stage) * len(edge_order)))
    G = G0.copy()
    G.remove_edges_from(edge_order[:n_remove])
    return G


def load_trajectory(stage: int) -> dict:
    d = np.load(S.TRAJ_DIR / f"traj_stage{stage:02d}_trial0.npz")
    return dict(V=d["V"], W=d["W"], dt=float(d["dt"]), stage=int(d["stage"]),
                K=float(d["K"]), sigma=float(d["sigma"]), a=float(d["a"]))


def detect_spikes(V, dt, threshold=0.5):
    above = V > threshold
    rising = np.diff(above.astype(np.int8), axis=0) == 1
    time_idx, neuron_idx = np.where(rising)
    return neuron_idx, time_idx * dt


def run_info() -> dict:
    return json.loads(S.RUN_INFO.read_text())


def stage_metrics() -> list[dict]:
    """Per-stage graph metrics of the canonical campaign."""
    return run_info()["metrics_per_stage"]


# --------------------------------------------------------------------------
# f_c estimators, copied from fhn_percolation.py
# --------------------------------------------------------------------------
def f_c_by_LCC_half(fractions, LCC_mean, N=S.N):
    rel = LCC_mean / N
    above = np.where(rel >= 0.5)[0]
    below = np.where(rel < 0.5)[0]
    if len(above) == 0 or len(below) == 0:
        return None
    i1, i2 = above[-1], below[0]
    if i2 - i1 != 1:
        return None
    f1, f2 = fractions[i1], fractions[i2]
    L1, L2 = rel[i1], rel[i2]
    return float(f1 + (f2 - f1) * (0.5 - L1) / (L2 - L1))


def f_c_by_steepest_drop(fractions, LCC_mean):
    return float(fractions[int(np.argmin(np.gradient(LCC_mean, fractions)))])


# --------------------------------------------------------------------------
# cached derived quantities
# --------------------------------------------------------------------------
def _cache_path(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def phase_distribution_data(stages=(1, 5, 10), n_snapshots=50, force=False):
    """Within-component phase deviations and size-weighted between-component
    pairwise mean-phase differences, computed exactly as in
    fhn_diagnostics.plot_phase_distributions."""
    path = _cache_path("phase_distributions.npz")
    if path.exists() and not force:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    v_star, w_star = fhn_fixed_point()
    out = {}
    for stage in stages:
        d = load_trajectory(stage)
        V, W = d["V"], d["W"]
        t_start = int(0.75 * V.shape[0])
        n_snap = min(n_snapshots, V.shape[0] - t_start)
        idxs = np.linspace(t_start, V.shape[0] - 1, n_snap).astype(int)

        G = reconstruct_graph_at_stage(stage)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        big = [np.array(sorted(c), dtype=np.int64) for c in comps if len(c) >= 2]

        within, pair_d, pair_w = [], [], []
        for ti in idxs:
            ph = np.arctan2(W[ti] - w_star, V[ti] - v_star)
            mph, sizes = [], []
            for comp in big:
                cp = ph[comp]
                m = np.angle(np.mean(np.exp(1j * cp)))
                mph.append(m)
                sizes.append(len(comp))
                within.append(np.angle(np.exp(1j * (cp - m))))
            if len(big) >= 2:
                mph = np.asarray(mph)
                sizes = np.asarray(sizes, dtype=float)
                diff = np.angle(np.exp(1j * (mph[:, None] - mph[None, :])))
                wgt = sizes[:, None] * sizes[None, :]
                iu = np.triu_indices(len(mph), k=1)
                pair_d.append(diff[iu])
                pair_w.append(wgt[iu])

        wv = np.concatenate(within)
        out[f"within_{stage}"] = wv
        fine = np.linspace(-np.pi, np.pi, 1257)       # 0.005 rad bins
        cnt, edg = np.histogram(wv, bins=fine, density=True)
        ctr = 0.5 * (edg[:-1] + edg[1:])
        hi = np.where(cnt >= 0.5 * cnt.max())[0]
        out[f"fwhm_{stage}"] = np.array(
            float(ctr[hi[-1]] - ctr[hi[0]]) if len(hi) else np.nan)
        out[f"ncomp_{stage}"] = np.array(len(big))
        if pair_d:
            out[f"pair_d_{stage}"] = np.concatenate(pair_d)
            out[f"pair_w_{stage}"] = np.concatenate(pair_w)

    np.savez_compressed(path, **out)
    return out


def spectra_data(stages=(1, 3, 5, 7, 10), force=False):
    """Welch mean-field spectra plus peak and integrated power per stage."""
    path = _cache_path("spectra.npz")
    if path.exists() and not force:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    out = {"stages": np.asarray(stages)}
    f_peak, p_peak, p_tot = [], [], []
    for stage in stages:
        d = load_trajectory(stage)
        mean_v = d["V"].mean(axis=1).astype(np.float64)
        dt = d["dt"]
        # 16384 resolves the fundamental at 1/54.6 time units. The
        # published setting of 8192 put the peak in the first bin above
        # zero and so reported a period of 82 rather than 55.
        nperseg = min(16384, len(mean_v) // 2)
        freqs, psd = welch(mean_v, fs=1.0 / dt, nperseg=nperseg)
        out[f"f_{stage}"] = freqs
        out[f"psd_{stage}"] = psd
        band = (freqs > 0.01) & (freqs < 0.5)
        k = int(np.argmax(psd[band]))
        f_peak.append(float(freqs[band][k]))
        p_peak.append(float(psd[band][k]))
        p_tot.append(float(np.trapezoid(psd, freqs)))
    out["f_peak"] = np.asarray(f_peak)
    out["psd_peak"] = np.asarray(p_peak)
    out["psd_total"] = np.asarray(p_tot)
    np.savez_compressed(path, **out)
    return out


def periodicity_data(stages=(1, 3, 5, 7, 10), max_lag=60.0, force=False):
    """Mean-field variance per stage and the normalised autocorrelation."""
    path = _cache_path("periodicity.npz")
    if path.exists() and not force:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    out = {"stages": np.asarray(stages)}
    variances = []
    for stage in stages:
        d = load_trajectory(stage)
        mean_v = d["V"].mean(axis=1).astype(np.float64)
        dt = d["dt"]
        variances.append(float(np.var(mean_v)))
        mv = mean_v - mean_v.mean()
        nfft = 2 ** int(np.ceil(np.log2(2 * len(mv))))
        f = np.fft.fft(mv, n=nfft)
        ac = np.fft.ifft(f * np.conj(f)).real[:len(mv)]
        ac /= ac[0]
        n_lag = min(len(ac), int(max_lag / dt))
        out[f"lag_{stage}"] = np.arange(n_lag) * dt
        out[f"ac_{stage}"] = ac[:n_lag]
    out["variance"] = np.asarray(variances)
    np.savez_compressed(path, **out)
    return out


def raster_data(stages=(1, 5, 10), t_window=150.0, n_show=200, force=False):
    """Spike times for the first n_show neurons and the mean field, over a
    common t_window so the three stages are directly comparable."""
    path = _cache_path("rasters.npz")
    if path.exists() and not force:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    out = {}
    for stage in stages:
        d = load_trajectory(stage)
        dt = d["dt"]
        n_steps = min(d["V"].shape[0], int(round(t_window / dt)))
        V = d["V"][:n_steps]
        n_idx, t_spk = detect_spikes(V, dt)
        keep = n_idx < n_show
        out[f"nidx_{stage}"] = n_idx[keep]
        out[f"tspk_{stage}"] = t_spk[keep]
        out[f"t_{stage}"] = np.arange(n_steps) * dt
        out[f"mv_{stage}"] = d["V"].mean(axis=1)[:n_steps]
    np.savez_compressed(path, **out)
    return out


def network_layouts(stages=(1, 5, 10), force=False):
    """Spring layouts and component colouring for the network panels."""
    path = _cache_path("layouts.npz")
    if path.exists() and not force:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    # one layout, computed on the healthy Stage-1 graph and reused at every
    # stage, so the same neuron sits at the same point in all three panels
    G1 = reconstruct_graph_at_stage(1)
    pos0 = nx.spring_layout(G1, seed=S.LAYOUT_SEED)

    out = {}
    for stage in stages:
        G = reconstruct_graph_at_stage(stage)
        pos = pos0
        nodes = np.array(sorted(G.nodes()))
        out[f"pos_{stage}"] = np.array([pos[n] for n in nodes])
        out[f"nodes_{stage}"] = nodes
        out[f"edges_{stage}"] = np.array(list(G.edges()), dtype=np.int64)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        cid = np.zeros(S.N, dtype=np.int64)
        for i, c in enumerate(comps):
            cid[list(c)] = i
        out[f"comp_{stage}"] = cid
        out[f"ncomp_{stage}"] = np.array(len(comps))
        out[f"lcc_{stage}"] = np.array(len(comps[0]))
    np.savez_compressed(path, **out)
    return out

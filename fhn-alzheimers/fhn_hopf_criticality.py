"""
Decides whether the single-neuron Hopf bifurcation is supercritical or
subcritical, by continuation sweeps across a_Hopf in both directions.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

B, EPS, I_EXT = 0.80, 0.06, 0.30
OUT = Path("alzheimers_hopf")


def fixed_point(a):
    def f(v):
        return v - v**3 / 3.0 + I_EXT - (v + a) / B
    grid = np.linspace(-3.0, 3.0, 601)
    sc = np.where(np.diff(np.sign(f(grid))))[0]
    v = brentq(f, grid[sc[0]], grid[sc[0] + 1])
    return v, (v + a) / B


def a_hopf():
    # (v*)^2 = 1 - eps*b  at the Hopf point
    return brentq(lambda a: fixed_point(a)[0] ** 2 - (1.0 - EPS * B), 0.4, 1.2)


def rk4_amplitude(a, v0, w0, dt=0.005, T=4000.0, frac=0.4):
    """Integrate the isolated neuron. Returns (amplitude, v_final, w_final), the
    amplitude being max-min of v over the last `frac` of the trace."""
    n = int(T / dt)
    keep = int(frac * n)
    vs = np.empty(keep)
    v, w = v0, w0

    def drift(v, w):
        return v - v**3 / 3.0 - w + I_EXT, EPS * (v + a - B * w)

    for k in range(n):
        k1v, k1w = drift(v, w)
        k2v, k2w = drift(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = drift(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = drift(v + dt * k3v, w + dt * k3w)
        v += dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        w += dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        if k >= n - keep:
            vs[k - (n - keep)] = v
    return float(vs.max() - vs.min()), v, w


def sweep(a_values, seed_state):
    """Continuation sweep: carry the final state into the next a."""
    amps = np.empty(len(a_values))
    v, w = seed_state
    for i, a in enumerate(a_values):
        amp, v, w = rk4_amplitude(a, v, w)
        amps[i] = amp
    return amps


def first_lyapunov_coefficient():
    """First Lyapunov coefficient l1 at the Hopf point, negative supercritical and
    positive subcritical. This is the test that settles it: the sweeps below
    corroborate but cannot decide, since a supercritical Hopf plus a fold of
    limit cycles also jumps. Formula from Guckenheimer & Holmes (3.4.11)."""
    aH = a_hopf()
    vs, _ = fixed_point(aH)
    J = np.array([[1.0 - vs**2, -1.0], [EPS, -EPS * B]])
    w0 = float(np.sqrt(np.linalg.det(J)))

    ev, evec = np.linalg.eig(J)
    q = evec[:, int(np.argmax(ev.imag))]
    q = q / q[0]
    T = np.column_stack([q.real, -q.imag])
    Ti = np.linalg.inv(T)

    c1, c2 = Ti[0, 0], Ti[1, 0]
    T11, T12 = T[0, 0], T[0, 1]
    F2, F3 = -2.0 * vs, -2.0            # d2F/dx2, d3F/dx3 at the fixed point

    f_uu, f_uv, f_vv = c1 * F2 * T11**2, c1 * F2 * T11 * T12, c1 * F2 * T12**2
    g_uu, g_uv, g_vv = c2 * F2 * T11**2, c2 * F2 * T11 * T12, c2 * F2 * T12**2
    f_uuu, f_uvv = c1 * F3 * T11**3, c1 * F3 * T11 * T12**2
    g_uuv, g_vvv = c2 * F3 * T11**2 * T12, c2 * F3 * T12**3

    l1 = (1.0 / 16.0) * (f_uuu + f_uvv + g_uuv + g_vvv) + (1.0 / (16.0 * w0)) * (
        f_uv * (f_uu + f_vv) - g_uv * (g_uu + g_vv) - f_uu * g_uu + f_vv * g_vv)
    return float(l1), w0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aH = a_hopf()
    vH, wH = fixed_point(aH)
    print(f"a_Hopf = {aH:.4f}   (v*)^2 = {vH**2:.4f}  (target {1-EPS*B:.4f})")

    # the bistable window is a few thousandths wide: step 0.0005 near a_Hopf,
    # 0.004 elsewhere
    COARSE, FINE, FINE_HALFWIDTH = 0.004, 0.0005, 0.02
    coarse = np.arange(0.600, 0.760 + 1e-9, COARSE)
    fine = np.arange(aH - FINE_HALFWIDTH, aH + FINE_HALFWIDTH + 1e-9, FINE)
    a_grid = np.unique(np.round(np.concatenate([coarse, fine]), 6))
    print(f"grid: {len(a_grid)} points, step {FINE} within "
          f"+-{FINE_HALFWIDTH} of a_Hopf, {COARSE} outside")

    # forward: start deep in the oscillatory regime, push a up
    v0, w0 = fixed_point(a_grid[0])
    amp_fwd = sweep(a_grid, (v0 + 0.3, w0))            # kick onto the cycle

    # backward: start on the stable side, pull a down
    v1, w1 = fixed_point(a_grid[-1])
    amp_bwd = sweep(a_grid[::-1], (v1 + 0.01, w1))[::-1]

    osc = 1e-3                                          # amplitude threshold
    fwd_on = a_grid[amp_fwd > osc]
    bwd_on = a_grid[amp_bwd > osc]
    # onset (forward, a increasing -> cycle dies) and revival (backward)
    a_die = fwd_on.max() if len(fwd_on) else float("nan")
    a_revive = bwd_on.max() if len(bwd_on) else float("nan")
    hysteresis = float(abs(a_revive - a_die))

    # amplitude jump at the forward death point
    idx_die = int(np.argmin(np.abs(a_grid - a_die)))
    amp_before_death = float(amp_fwd[max(idx_die, 0)])

    # supercritical scaling test: amp^2 vs (aH - a) just below onset
    mask = (a_grid < aH) & (a_grid > aH - 0.05) & (amp_fwd > osc)
    scaling = {}
    if mask.sum() >= 3:
        x = aH - a_grid[mask]
        y = amp_fwd[mask] ** 2
        slope, intercept = np.polyfit(x, y, 1)
        ss = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
        scaling = dict(amp2_vs_dist_slope=float(slope),
                       amp2_vs_dist_intercept=float(intercept),
                       amp2_vs_dist_r2=float(ss))

    # verdict: the analytic l1 decides; the sweeps corroborate
    l1, w0 = first_lyapunov_coefficient()
    verdict = "subcritical" if l1 > 0 else "supercritical"

    # judged against the fine step, which is what resolves the window
    hyst_significant = hysteresis > 3 * FINE
    jump_significant = amp_before_death > 0.5      # near-full-amplitude cycle at death
    numerics_agree = bool((hyst_significant or jump_significant) == (l1 > 0))

    print(f"\nforward cycle dies at   a ~ {a_die:.4f}  (amp just before = {amp_before_death:.3f})")
    print(f"backward cycle revives at a ~ {a_revive:.4f}")
    print(f"hysteresis width = {hysteresis:.4f}  "
          f"({'significant' if hyst_significant else 'none'})")
    if scaling:
        print(f"amp^2 vs (a_Hopf - a): slope={scaling['amp2_vs_dist_slope']:.3f}, "
              f"R^2={scaling['amp2_vs_dist_r2']:.3f}")
    print(f"\nfirst Lyapunov coefficient l1 = {l1:+.6f}   (omega0 = {w0:.6f})")
    print(f"  l1 > 0 => subcritical, l1 < 0 => supercritical")
    print(f"  numerical sweeps agree with l1: {numerics_agree}")
    print(f"\nVERDICT: the Hopf bifurcation is {verdict.upper()}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(a_grid, amp_fwd, "o-", color="C0", ms=4, lw=1.6,
            label="forward (a increasing)")
    ax.plot(a_grid, amp_bwd, "s--", color="C3", ms=4, lw=1.6,
            label="backward (a decreasing)")
    ax.axvline(aH, color="k", ls=":", lw=1.2, label=f"$a_{{Hopf}}$ = {aH:.3f}")
    ax.set_xlabel("bifurcation parameter  a")
    ax.set_ylabel("limit-cycle amplitude  (max$-$min of v)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "hopf_criticality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = dict(a_hopf=float(aH), v_star_hopf=float(vH),
               verdict=verdict,
               first_lyapunov_coefficient=l1,
               omega0=w0,
               criticality_decided_by="first Lyapunov coefficient",
               numerics_agree_with_l1=numerics_agree,
               fine_step=FINE,
               forward_cycle_death_a=float(a_die),
               backward_cycle_revival_a=float(a_revive),
               hysteresis_width=hysteresis,
               hysteresis_significant=bool(hyst_significant),
               amp_just_before_death=amp_before_death,
               amp_jump_significant=bool(jump_significant),
               scaling=scaling,
               a_grid=[float(x) for x in a_grid],
               amp_forward=[float(x) for x in amp_fwd],
               amp_backward=[float(x) for x in amp_bwd])
    with open(OUT / "hopf_criticality.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\noutputs in {OUT.resolve()}")


if __name__ == "__main__":
    main()

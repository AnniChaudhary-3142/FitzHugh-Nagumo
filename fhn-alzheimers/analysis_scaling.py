"""
Fokker-Planck ratio table, the sigma^2/K power-law fit, and the R_k
size-dependence check. Reads runs/ only and simulates nothing.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
CONDITIONS = ["K020", "canonical", "K050", "s002", "s010"]


def load_condition(name):
    d = json.loads((RUNS / name / "out" / "robustness_summary.json").read_text())
    return (d["config"]["K"], d["config"]["sigma"],
            d["stats_R_global"]["stage1_mean"])


def fokker_planck_table():
    rows = []
    for name in CONDITIONS:
        K, sigma, R = load_condition(name)
        rows.append((name, K, sigma, R, (1.0 - R) * K / sigma**2))

    print("Fokker-Planck ratio (1-R) K / sigma^2 at Stage 1")
    print(f"{'run':<10}{'K':>7}{'sigma':>8}{'R_1':>10}{'ratio':>9}")
    for name, K, sigma, R, ratio in rows:
        print(f"{name:<10}{K:>7.2f}{sigma:>8.2f}{R:>10.4f}{ratio:>9.2f}")

    ratios = np.array([r[4] for r in rows])
    mean, sd = ratios.mean(), ratios.std(ddof=1)
    print(f"\nmean {mean:.2f} +- {sd:.2f}, coefficient of variation "
          f"{100 * sd / mean:.0f}%")
    print("A constant ratio would confirm 1-R ~ sigma^2/K.")
    return rows


def _loglog_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = np.sqrt((resid**2).sum() / (len(x) - 2) / ((x - x.mean())**2).sum())
    r2 = 1.0 - (resid**2).sum() / ((y - y.mean())**2).sum()
    return slope, se, r2, float(np.exp(intercept))


def power_law_fit(rows):
    """Fit on the five condition means and on all ninety Stage 1 network-trials.
    The manuscript quotes the second."""
    print("\n1-R = c (sigma^2/K)^m, predicted m = 1")

    x = np.log([sigma**2 / K for _, K, sigma, _, _ in rows])
    y = np.log([1.0 - R for _, _, _, R, _ in rows])
    m, se, r2, c = _loglog_fit(x, y)
    print(f"  condition means   m = {m:.3f} +- {se:.3f}   "
          f"c = {c:.3f}   R^2 = {r2:.4f}   n = {len(x)}")

    xs, ys = [], []
    for name, K, sigma, _, _ in rows:
        R_net = np.load(RUNS / name / "out" / "robustness_full_data.npz")["R_global"]
        stage1 = np.asarray(R_net)[:, 0].ravel()
        xs.extend([np.log(sigma**2 / K)] * len(stage1))
        ys.extend(np.log(1.0 - stage1))
    m_p, se_p, r2_p, c_p = _loglog_fit(np.array(xs), np.array(ys))
    print(f"  per network       m = {m_p:.3f} +- {se_p:.3f}   "
          f"c = {c_p:.3f}   R^2 = {r2_p:.4f}   n = {len(xs)}")
    return m, se


def component_size_dependence():
    path = RUNS / "canonical" / "out" / "per_component_per_network.json"
    per_network = json.loads(path.read_text())

    sizes, R_k = [], []
    for network in per_network:
        for stage in network:
            for n, R in zip(stage["sizes"], stage["R_k"]):
                if n >= 2:
                    sizes.append(n)
                    R_k.append(R)
    sizes = np.asarray(sizes, dtype=float)
    R_k = np.asarray(R_k, dtype=float)
    rho = float(np.corrcoef(np.log(sizes), R_k)[0, 1])

    print(f"\nComponents of size >= 2 pooled over networks and stages: {len(sizes)}")
    print(f"corr(log n_k, R_k) = {rho:.3f}")
    print("R_k is not size-independent, so the single-value approximation "
          "R_k = R_c carries a bias.")
    return rho


def main():
    rows = fokker_planck_table()
    power_law_fit(rows)
    component_size_dependence()


if __name__ == "__main__":
    main()

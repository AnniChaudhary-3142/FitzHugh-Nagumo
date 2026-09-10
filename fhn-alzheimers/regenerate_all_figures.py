"""
Rebuilds every paper figure from stored data. Needs the trajectory .npz
files, which are not distributed with this repository.
"""
import json
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")

import fhn_alzheimers as AZ
import fhn_diagnostics as D
import replot_figures as RP

DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures_normalized")
TRAJ = Path("alzheimers_results/trajectories_canonical")
LAYOUT_SEED = 20260529


def load_traj_npz(stage):
    d = np.load(TRAJ / f"traj_stage{stage:02d}_trial0.npz")
    return dict(V=d["V"], W=d["W"], dt=float(d["dt"]), stage=int(d["stage"]),
                K=float(d["K"]), sigma=float(d["sigma"]), a=float(d["a"]))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cfg = AZ.Config()

    # ---- 1. the four stored-npz figures (titles already stripped in RP) ----
    RP.main(DEST)
    print("replot figures done")

    # ---- 2. diagnostics: phase_distributions, spectra, periodicity ----
    diag_stages = [1, 3, 5, 7, 10]
    stages_data = {s: D.load_trajectory(s) for s in diag_stages}
    D.plot_phase_distributions(stages_data, DEST / "phase_distributions.png")
    D.plot_mean_field_spectra(stages_data, DEST / "mean_field_spectra.png")
    D.plot_periodicity_diagnostics(stages_data, DEST / "periodicity_diagnostics.png")
    print("diagnostics figures done")

    # ---- 3. fhn_alzheimers: phase_portrait, graph_metrics, rasters, networks ----
    AZ.plot_phase_portrait(cfg, DEST / "phase_portrait.png")

    metrics = [json.loads(str(m)) for m in
               np.load("alzheimers_results/results.npz", allow_pickle=True)["metrics"]]
    AZ.plot_graph_metrics(metrics, cfg, DEST / "graph_metrics.png")

    for stage in (1, 5, 10):
        m = metrics[stage - 1]
        tr = load_traj_npz(stage)
        AZ.plot_raster(tr["V"], stage, cfg, m, DEST / f"raster_stage{stage}.png")
        G = D.reconstruct_graph_at_stage(stage)
        pos = nx.spring_layout(G, seed=LAYOUT_SEED)
        AZ.plot_network(G, stage, cfg, m, pos, DEST / f"network_stage{stage}.png")
    print("alzheimers figures done")
    print("all figures regenerated into", DEST.resolve())


if __name__ == "__main__":
    main()

"""
Rebuilds the three figures that need no stored trajectories, straight
from runs/.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

from fhn_alzheimers import Config
import replot_figures as RP

HERE = Path(__file__).resolve().parent
DEST = HERE / "figures_normalized"
DEST.mkdir(parents=True, exist_ok=True)

d = np.load(HERE / "runs/canonical/out/robustness_full_data.npz")
cfg = Config(); cfg.ablation_strategy = "betweenness"
RP.plot_multi_network(d["R_global"], d["R_component"], cfg,
                       DEST / "R_multi_network_robustness.png")
print("wrote R_multi_network_robustness.png")

d = np.load(HERE / "runs/random/out/robustness_full_data.npz")
cfg = Config(); cfg.ablation_strategy = "random"
RP.plot_multi_network(d["R_global"], d["R_component"], cfg,
                       DEST / "R_multi_network_random.png")
print("wrote R_multi_network_random.png")

d = np.load(HERE / "runs/percolation/out/percolation_data.npz")
RP.plot_percolation_diagram(d, Config(), DEST / "percolation_diagram.png")
print("wrote percolation_diagram.png")

print("\nAll 3 static figures written to", DEST.resolve())

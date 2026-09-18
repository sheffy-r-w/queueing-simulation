"""
Export every deck figure without the baked-in headline and subtitle, to figures/slides/.

The slide title carries the finding, so the headline would duplicate it. Colours,
legends, panel titles and axis labels are unchanged. Re-plots from the existing
CSVs (no simulation); the rank-hull timing is re-measured (a few seconds).

Run:  python -m experiments.slide_export
"""

import os
import subprocess
import sys

STEPS = [
    ["-m", "experiments.slide_figures"],
    ["-m", "experiments.eda_tails"],
    ["-m", "experiments.rank_hull"],
    ["-m", "experiments.baseline_comparison", "--plot"],
    ["-m", "experiments.kupdating_drift", "--plot"],
    ["-m", "experiments.imitation", "overlay"],
    ["-m", "experiments.rl", "--plot"],
]

if __name__ == "__main__":
    env = dict(os.environ, EGITTINS_SLIDES="1")
    root = os.path.join(os.path.dirname(__file__), "..")
    for step in STEPS:
        print("+", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], cwd=root, env=env, check=True)
    print("slide exports are in figures/slides/")

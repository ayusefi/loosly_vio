#!/usr/bin/env python3
"""Draws the dead-reckoning drift against the filter's own 3-sigma envelope.

    ./scripts/plot_drift.py drift.csv out.png [title]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 1

    data = np.loadtxt(sys.argv[1], delimiter=",", comments="#")
    seconds, drift, three_sigma = data[:, 0], data[:, 1], data[:, 2]

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.fill_between(seconds, 0, three_sigma, alpha=0.25,
                      label="filter's own 3-sigma")
    axes.plot(seconds, three_sigma, linewidth=1.0)
    axes.plot(seconds, drift, linewidth=1.6, label="actual drift")

    axes.set_yscale("log")
    axes.set_xlabel("seconds")
    axes.set_ylabel("position error (m)")
    axes.set_title(sys.argv[3] if len(sys.argv) == 4 else "IMU dead reckoning")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()

    covered = np.mean(drift <= three_sigma) * 100.0
    axes.text(0.02, 0.95, f"inside the envelope: {covered:.1f}% of the time",
              transform=axes.transAxes, va="top")

    figure.tight_layout()
    figure.savefig(sys.argv[2], dpi=130)
    print(f"wrote {sys.argv[2]}  ({covered:.1f}% inside)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
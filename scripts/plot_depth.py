#!/usr/bin/env python3
"""Draws stereo depth uncertainty against depth, with the measured depths behind it.

    ./scripts/plot_depth.py depth.csv out.png [max_depth]
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
    depth, sigma = data[:, 0], data[:, 1]
    max_depth = float(sys.argv[3]) if len(sys.argv) == 4 else 17.0

    order = np.argsort(depth)
    depth, sigma = depth[order], sigma[order]
    relative = 100.0 * sigma / depth

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    top.hist(depth, bins=120, color="tab:blue", alpha=0.7)
    top.axvline(max_depth, color="tab:red", linestyle="--",
                label="max_depth = %.0f m" % max_depth)
    top.set_ylabel("features triangulated")
    top.set_title("what the camera actually sees, and how far we can trust it")
    top.legend()
    top.grid(alpha=0.3)

    bottom.plot(depth, relative, color="tab:orange",
                label="relative uncertainty, sigma_z / z")
    bottom.axhline(10.0, color="tab:grey", linestyle=":", label="10% line")
    bottom.axvline(max_depth, color="tab:red", linestyle="--")
    bottom.set_xlabel("depth (m)")
    bottom.set_ylabel("sigma_z / z  (%)")
    bottom.set_ylim(0, 25)
    bottom.legend()
    bottom.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(sys.argv[2], dpi=130)
    print("wrote %s" % sys.argv[2])

    inside = np.count_nonzero(depth <= max_depth)
    print("depths within max_depth: %d of %d = %.2f%%"
          % (inside, len(depth), 100.0 * inside / len(depth)))
    print("median relative uncertainty: %.2f%%" % np.median(relative))
    return 0


if __name__ == "__main__":
    sys.exit(main())
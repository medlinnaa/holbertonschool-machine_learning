#!/usr/bin/env python3
"""2-change_scale.py: Plot exponential decay of C-14 on a log scale."""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Display C-14 exponential decay with logarithmic y-axis."""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y)
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.yscale('log')        # logarithmic y-axis
    plt.xlim(0, 28650)       # x-axis range
    plt.show()

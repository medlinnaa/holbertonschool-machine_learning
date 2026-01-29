#!/usr/bin/env python3
"""0-line.py: Plot y = x^3 as a red line graph from x=0 to x=10."""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot y = x^3 as a solid red line with x-axis from 0 to 10."""
    x = np.arange(0, 11)
    y = x ** 3

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, "r-")
    plt.xlim(0, 10)
    plt.show()

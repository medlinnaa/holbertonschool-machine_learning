#!/usr/bin/env python3
"""4-array.py"""

import numpy as np


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and returns them as a numpy.ndarray.
    """
    return df[["High", "Close"]].tail(10).to_numpy()

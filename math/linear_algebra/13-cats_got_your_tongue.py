#!/usr/bin/env python3
"""concatenating two numpy arrays along a given axis."""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """a function that returns
    the concatenation of two numpy.ndarrays along axis."""
    return np.concatenate((mat1, mat2), axis=axis)

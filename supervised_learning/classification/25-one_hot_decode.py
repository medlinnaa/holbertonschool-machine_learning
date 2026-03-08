#!/usr/bin/env python3
"""Contains the one_hot_decode function"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels
    one_hot: numpy.ndarray with shape (classes, m)
    Returns: numpy.ndarray with shape (m,) containing numeric labels,
             or None on failure
    """
    # 1. Validation: Must be a numpy array with 2 dimensions
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        # 2. Find the index of the maximum value along the rows (axis 0)
        # This effectively 'squashes' the classes back into a single vector
        # Labels are retrieved by finding where the 1 is located
        decoded = np.argmax(one_hot, axis=0)

        return decoded
    except Exception:
        return None

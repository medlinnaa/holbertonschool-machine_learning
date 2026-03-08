#!/usr/bin/env python3
"""Contains the one_hot_encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Y: numpy.ndarray with shape (m,) containing numeric class labels
    classes: maximum number of classes found in Y
    Returns: one-hot encoding of Y with shape (classes, m), or None on failure
    """
    # Check if Y is a valid numpy array and not empty
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None

    # Check if classes is an integer and large enough to hold the labels
    if not isinstance(classes, int):
        return None

    try:
        m = Y.shape[0]
        # Create a matrix filled with zeros
        one_hot = np.zeros((classes, m))

        # Select the specific row (Y) and column (0 to m-1) to set to 1
        # This uses indexing instead of a loop
        one_hot[Y, np.arange(m)] = 1

        return one_hot
    except Exception:
        # Return None if the labels in Y exceed the number of classes
        return None

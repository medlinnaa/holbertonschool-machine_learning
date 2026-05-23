#!/usr/bin/env python3
"""
Module calculating the total intra-cluster variance of a dataset.
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        C (numpy.ndarray): The centroid means, shape (k, d).

    Returns:
        float: The total variance, or None on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None

    # Ensure dimensions match between dataset and centroids
    if X.shape[1] != C.shape[1]:
        return None

    # Broadcast shapes to calculate squared distance efficiently
    # X shape: (n, 1, d) -> diff shape: (n, k, d)
    # dist_sq shape becomes (n, k) after summing along the features axis
    dist_sq = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

    # Find the squared distance to the closest centroid
    min_dist_sq = np.min(dist_sq, axis=1)

    # The total variance is the sum of these minimum squared distances
    var = np.sum(min_dist_sq)

    return float(var)

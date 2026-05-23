#!/usr/bin/env python3
"""
Module to test for the optimum number of clusters by variance.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check (inclusive).
        kmax (int): Maximum number of clusters to check (inclusive).
        iterations (int): Maximum number of iterations per K-means run.

    Returns:
        tuple: (results, d_vars) or (None, None) on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None

    # If kmax is not provided, set it to the total number of data points
    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax <= 0:
        return None, None

    # We must analyze at least 2 different cluster sizes
    if kmin >= kmax:
        return None, None

    results = []
    d_vars = []
    var_min = None

    # Loop over the cluster sizes
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        # Append the K-means output tuple
        results.append((C, clss))

        # Calculate the variance of the current cluster size
        var = variance(X, C)

        # Save the baseline variance from the smallest cluster size
        if var_min is None:
            var_min = var

        # Track the difference from the smallest cluster size
        d_vars.append(var_min - var)

    return results, d_vars

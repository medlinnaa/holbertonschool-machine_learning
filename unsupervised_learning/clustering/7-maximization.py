#!/usr/bin/env python3
"""
Module calculating the maximization step in the EM algorithm.
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm of a GMM.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        g (numpy.ndarray): The posterior probabilities, shape (k, n).

    Returns:
        tuple: (pi, m, S) or (None, None, None) on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    # Check if posterior probabilities sum to 1 across clusters
    # Using np.isclose to prevent floating point precision errors
    sums = np.sum(g, axis=0)
    valid_sums = np.ones(X.shape[0])
    if not np.all(np.isclose(sums, valid_sums)):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # Nk is the effective number of points assigned to each cluster
    Nk = np.sum(g, axis=1)

    # Update the priors (pi)
    pi = Nk / n

    # Update the centroid means (m) using matrix multiplication
    # g is (k, n) and X is (n, d), resulting in a (k, d) matrix
    m = np.matmul(g, X) / Nk[:, np.newaxis]

    # Initialize the updated covariance matrices
    S = np.zeros((k, d, d))

    # Calculate the covariance matrices updating each cluster one by one
    for j in range(k):
        # Difference between data points and the updated mean
        diff = X - m[j]

        # Apply the posterior probability weights to the differences
        weighted_diff = g[j, :, np.newaxis] * diff

        # Calculate the outer product and
        # divide by the effective number of points
        S[j] = np.matmul(weighted_diff.T, diff) / Nk[j]

    return pi, m, S

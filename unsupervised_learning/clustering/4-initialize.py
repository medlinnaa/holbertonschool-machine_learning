#!/usr/bin/env python3
"""
Module initializing variables of a Gaussian Mixture Model.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables of a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        k (int): The number of clusters.

    Returns:
        tuple: (pi, m, S) or (None, None, None) on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape

    # Initialize priors evenly (summing to 1)
    pi = np.ones(k) / k

    # Initialize centroid means using our K-means algorithm
    m, _ = kmeans(X, k)

    # Initialize covariance matrices as identity matrices using tile
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S

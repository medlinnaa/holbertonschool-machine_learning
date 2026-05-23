#!/usr/bin/env python3
"""
Module for initializing cluster centroids for K-means.
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d) used for K-means.
        k (int): The number of clusters.

    Returns:
        numpy.ndarray: Shape (k, d) containing initialized centroids.
        None: On failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    _, d = X.shape

    # Calculate the minimum and maximum values along each dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Generate centroids using a uniform distribution and broadcasting
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids

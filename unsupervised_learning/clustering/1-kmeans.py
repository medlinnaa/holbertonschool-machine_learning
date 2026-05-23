#!/usr/bin/env python3
"""
Module for performing K-means clustering algorithm.
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d) used for K-means.
        k (int): The number of clusters.
        iterations (int): The maximum number of iterations.

    Returns:
        tuple: (C, clss) or (None, None) on failure.
            C (numpy.ndarray): Centroid means for each cluster, shape (k, d).
            clss (numpy.ndarray): Index of the cluster each point belongs to.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # First use of np.random.uniform: Initialize cluster centroids
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for _ in range(iterations):
        C_copy = np.copy(C)

        # Calculate distances using broadcasting
        # X shape: (n, d) -> (n, 1, d)
        # C shape: (k, d) -> (1, k, d) (Implicitly broadcasted by numpy)
        # Difference results in (n, k, d),
        # taking norm along axis 2 gives (n, k)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # Assign points to the closest cluster
        clss = np.argmin(distances, axis=1)

        # Update centroid means
        for j in range(k):
            cluster_points = X[clss == j]

            # Handle empty clusters by reinitializing
            if len(cluster_points) == 0:
                # Second use of np.random.uniform: Reinitialize empty centroid
                C[j] = np.random.uniform(
                    low=min_vals, high=max_vals, size=(1, d))[0]
            else:
                C[j] = cluster_points.mean(axis=0)

        # If centroids do not change, converge early
        if np.all(C_copy == C):
            return C, clss

    # If the maximum iterations run out, calculate assignments one last time
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss

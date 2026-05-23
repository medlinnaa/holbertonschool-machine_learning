i#!/usr/bin/env python3
"""
Module to perform agglomerative clustering on a dataset.
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        dist (float): Maximum cophenetic distance for all clusters.

    Returns:
        numpy.ndarray: Shape (n,) containing cluster indices of each point.
    """
    # Calculate the hierarchical linkage matrix using Ward's method
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Generate and display the dendrogram with the specified color threshold
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    # Flatten the hierarchy into discrete clusters based on the distance
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss

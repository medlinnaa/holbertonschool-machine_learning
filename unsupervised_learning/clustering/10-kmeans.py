#!/usr/bin/env python3
"""
Module to perform K-means clustering using scikit-learn.
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        k (int): The number of clusters.

    Returns:
        tuple: (C, clss)
            C (numpy.ndarray): Centroid means for each cluster, shape (k, d).
            clss (numpy.ndarray): Index of the cluster each point belongs to.
    """
    # Initialize the KMeans model with the specified number of clusters
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)

    # Fit the model to the dataset
    kmeans_model.fit(X)

    # Extract the cluster centers and the labels
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss

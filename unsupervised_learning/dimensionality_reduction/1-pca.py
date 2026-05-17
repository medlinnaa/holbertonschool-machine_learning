#!/usr/bin/env python3
"""
Module to perform PCA on a dataset and reduce its dimensionality
"""
import numpy as np


def pca(X, ndim):
    """
    Performs Principal Component Analysis (PCA) on a dataset to transform it
    to a specified number of dimensions.

    Args:
        X (numpy.ndarray): dataset of shape (n, d).
            n is the number of data points.
            d is the number of dimensions in each point.
        ndim (int): the new dimensionality of the transformed X.

    Returns:
        numpy.ndarray: T, a matrix of shape (n, ndim) containing the
            transformed version of X.
    """
    # Center the dataset by subtracting the mean of each dimension
    X_centered = X - np.mean(X, axis=0)
    
    # Perform Singular Value Decomposition on the centered data
    _, _, Vh = np.linalg.svd(X_centered)
    
    # Extract the top 'ndim' principal components (rows of Vh) and transpose
    W = Vh[:ndim].T
    
    # Project the centered data onto the new dimensional space
    T = np.matmul(X_centered, W)
    
    return T

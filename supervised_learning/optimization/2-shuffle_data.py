#!/usr/bin/env python3
"""Module to shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
        Y: numpy.ndarray of shape (m, ny) to shuffle
    Returns: The shuffled X and Y matrices
    """
    # Create a random permutation of indices from 0 to m-1
    permutation = np.random.permutation(X.shape[0])

    # Reorder X and Y using the same permutation
    return X[permutation], Y[permutation]

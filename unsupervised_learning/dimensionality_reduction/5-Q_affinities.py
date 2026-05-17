#!/usr/bin/env python3
"""Module to calculate the Q affinities for t-SNE"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities.

    Args:
        Y (numpy.ndarray): shape (n, ndim) containing the low dimensional
            transformation of X
            n is the number of points
            ndim is the new dimensional representation of X

    Returns:
        (Q, num)
        Q: a numpy.ndarray of shape (n, n) containing the Q affinities
        num: a numpy.ndarray of shape (n, n) containing the numerator
            of the Q affinities
    """
    # Calculate squared pairwise distances using vectorized operations
    sum_Y = np.sum(np.square(Y), axis=1)
    D = sum_Y + sum_Y.reshape(-1, 1) - 2 * np.dot(Y, Y.T)

    # Clean up minor floating point inaccuracies (prevent negative distances)
    D = np.maximum(D, 0)

    # Calculate the numerator: (1 + ||yi - yj||^2)^(-1)
    num = 1.0 / (1.0 + D)

    # Set the diagonal elements to 0 (t-SNE defines affinities to oneself as 0)
    np.fill_diagonal(num, 0)

    # Calculate Q by dividing the numerator matrix
    # by the sum of all its elements
    Q = num / np.sum(num)

    return Q, num

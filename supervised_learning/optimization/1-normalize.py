#!/usr/bin/env python3
"""Module to normalize a matrix"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
        m: numpy.ndarray of shape (nx,) containing the mean of all features
        s: numpy.ndarray of shape (nx,) containing the std dev of all features
    Returns: The normalized X matrix
    """
    # use broadcasting to subtract mean and divide by standard deviation
    return (X - m) / s

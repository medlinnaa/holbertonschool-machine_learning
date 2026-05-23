#!/usr/bin/env python3
"""
Module calculating the probability density function of a Gaussian distribution.
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X (numpy.ndarray): Data points, shape (n, d).
        m (numpy.ndarray): Mean of the distribution, shape (d,).
        S (numpy.ndarray): Covariance of the distribution, shape (d, d).

    Returns:
        numpy.ndarray: PDF values, shape (n,), or None on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    # Verify dimensional alignment
    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    # Calculate the denominator of the PDF equation
    det_S = np.linalg.det(S)
    denominator = np.sqrt(((2 * np.pi) ** d) * det_S)

    # Calculate the exponent of the PDF equation without loops or diag()
    # (X - m) shape is (n, d). S_inv shape is (d, d).
    S_inv = np.linalg.inv(S)
    diff = X - m

    # diff * matmul(...) executes element-wise multiplication
    # sum(..., axis=1) collapses it horizontally, producing shape (n,)
    exponent = -0.5 * np.sum(diff * np.matmul(diff, S_inv), axis=1)

    # Calculate final probabilities
    P = np.exp(exponent) / denominator

    # Ensure no value drops below 1e-300
    P = np.maximum(P, 1e-300)

    return P

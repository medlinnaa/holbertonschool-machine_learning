#!/usr/bin/env python3
"""Module to calculate the gradients of Y for t-SNE"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): shape (n, ndim) containing the low dimensional
            transformation of X
        P (numpy.ndarray): shape (n, n) containing the P affinities of X

    Returns:
        (dY, Q)
        dY: a numpy.ndarray of shape (n, ndim) containing the gradients of Y
        Q: a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """
    # Get Q affinities and the numerator (1 + ||yi - yj||^2)^-1
    Q, num = Q_affinities(Y)

    # Calculate the weighted difference: (Pij - Qij) * num_ij
    # This acts as our coefficient matrix 'M'
    M = (P - Q) * num

    # Vectorize the gradient calculation:
    # dYi = sum_j(M_ij * (Yi - Yj))
    # dYi = Yi * sum_j(M_ij) - sum_j(M_ij * Yj)

    # Calculate sum of rows for M, keeping dimensions for broadcasting
    sum_M = np.sum(M, axis=1, keepdims=True)

    # Calculate dY using matrix multiplication to avoid loops
    dY = (sum_M * Y) - np.dot(M, Y)

    return dY, Q

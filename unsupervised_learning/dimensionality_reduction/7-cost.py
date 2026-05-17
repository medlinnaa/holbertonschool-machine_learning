#!/usr/bin/env python3
"""Module to calculate the cost of the t-SNE transformation"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation (Kullback-Leibler divergence).

    Args:
        P (numpy.ndarray): shape (n, n) containing the P affinities
        Q (numpy.ndarray): shape (n, n) containing the Q affinities

    Returns:
        C (float): the cost of the transformation
    """
    # Protect against division by 0 and log(0) errors by clipping the lower bound
    P_safe = np.maximum(P, 1e-12)
    Q_safe = np.maximum(Q, 1e-12)

    # Calculate the Kullback-Leibler divergence
    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C

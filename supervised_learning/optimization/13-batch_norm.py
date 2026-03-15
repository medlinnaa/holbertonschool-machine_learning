#!/usr/bin/env python3
"""Module to implement Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization
    Args:
        Z: numpy.ndarray of shape (m, n) to be normalized
        gamma: numpy.ndarray of shape (1, n) containing the scales
        beta: numpy.ndarray of shape (1, n) containing the offsets
        epsilon: small number to avoid division by zero
    Returns: the normalized Z matrix
    """
    # 1. Calculate Mean and Variance across the batch (axis=0)
    mu = np.mean(Z, axis=0)
    sigma_sq = np.var(Z, axis=0)

    # 2. Normalize Z
    Z_hat = (Z - mu) / np.sqrt(sigma_sq + epsilon)

    # 3. Scale and Shift (the learnable transformation)
    # Result = gamma * Z_normalized + beta
    return gamma * Z_hat + beta

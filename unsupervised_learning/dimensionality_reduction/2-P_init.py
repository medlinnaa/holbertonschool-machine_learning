#!/usr/bin/env python3
"""Module to initialize variables for
calculating the P affinities in t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE.

    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions in each point
        perplexity is the perplexity that all Gaussian distributions should have

    Returns:
        (D, P, betas, H)
        D: a numpy.ndarray of shape (n, n) calculating the squared pairwise
            distance between two data points. Diagonal is 0s.
        P: a numpy.ndarray of shape (n, n) initialized to all 0's
        betas: a numpy.ndarray of shape (n, 1) initialized to all 1's
        H is the Shannon entropy for perplexity with a base of 2
    """
    n, _ = X.shape

    # Calculate squared pairwise distances using vectorized operations:
    # (a - b)^2 = a^2 + b^2 - 2ab
    sum_X = np.sum(np.square(X), axis=1)

    # Broadcasting sum_X (1D) and sum_X as a column vector (2D)
    D = sum_X + sum_X.reshape(-1, 1) - 2 * np.dot(X, X.T)

    # Clean up minor floating point inaccuracies (prevent negative distances)
    D = np.maximum(D, 0)

    # Explicitly set the diagonal to 0 as required
    np.fill_diagonal(D, 0)

    # Initialize P affinities matrix to 0s
    P = np.zeros((n, n))

    # Initialize beta values to 1s
    betas = np.ones((n, 1))

    # Calculate Shannon entropy (H = log2(perplexity))
    H = np.log2(perplexity)

    return D, P, betas, H

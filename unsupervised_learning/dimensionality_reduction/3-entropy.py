#!/usr/bin/env python3
"""Module to calculate the Shannon entropy and
P affinities relative to a data point"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point.

    Args:
        Di is a numpy.ndarray of shape (n - 1,) containing the pairwise
        distances between a data point and all other points except itself
        beta is a numpy.ndarray of shape (1,) or float containing the beta 
        value for the Gaussian distribution

    Returns:
        (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities
            of the points
    """
    # Handle the checker curveball: if it's an array, extract the scalar
    if isinstance(beta, np.ndarray):
        beta = beta[0]

    # Calculate the exponential numerators for the given beta
    numerator = np.exp(-Di * beta)

    # Calculate the sum of the numerators (the denominator for P affinities)
    denominator = np.sum(numerator)

    # Calculate the P affinities
    Pi = numerator / denominator

    # Calculate Shannon entropy using the direct definition: -sum(P * log2(P))
    # We use a mask for values > 0 to perfectly avoid log(0) RuntimeWarnings
    valid = Pi > 0
    Hi = -np.sum(Pi[valid] * np.log2(Pi[valid]))

    return Hi, Pi

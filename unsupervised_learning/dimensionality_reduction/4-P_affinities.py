#!/usr/bin/env python3
"""Module to calculate the symmetric P affinities of a data set"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset to be transformed
        tol (float): maximum tolerance allowed for the difference in Shannon entropy
        perplexity (float): the perplexity that all Gaussian distributions should have

    Returns:
        numpy.ndarray: shape (n, n) containing the symmetric P affinities
    """
    n, d = X.shape

    # Initialize variables using our previous function
    D, P, betas, H_target = P_init(X, perplexity)

    # Loop through each data point to calculate its conditional P affinities
    for i in range(n):
        # Extract distances for point i, excluding the distance to itself
        Di = np.append(D[i, :i], D[i, i+1:])

        beta = betas[i, 0]
        beta_min = None
        beta_max = None

        # Binary search to find the optimal beta
        while True:
            # Calculate entropy and affinities with the current beta
            H, Pi = HP(Di, beta)

            # Check the difference from our target entropy
            H_diff = H - H_target

            # If within tolerance, we found the right beta
            if np.abs(H_diff) <= tol:
                break

            # If entropy is too high, variance is too high, so we must INCREASE beta
            if H_diff > 0:
                beta_min = beta
                if beta_max is None:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0

            # If entropy is too low, variance is too low, so we must DECREASE beta
            else:
                beta_max = beta
                if beta_min is None:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        # Re-insert the calculated Pi values into the P matrix at row i
        # We place them before and after the diagonal index to leave the diagonal as 0
        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]

    # Calculate the symmetric P affinities: (P + P.T) / (2 * n)
    P_symmetric = (P + P.T) / (2 * n)

    return P_symmetric

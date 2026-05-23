#!/usr/bin/env python3
"""
Module calculating the expectation step in the EM algorithm.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm of a GMM.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        pi (numpy.ndarray): The priors of each cluster, shape (k,).
        m (numpy.ndarray): Centroid means of each cluster, shape (k, d).
        S (numpy.ndarray): Covariance matrices, shape (k, d, d).

    Returns:
        tuple: (g, l) or (None, None) on failure.
            g is a numpy.ndarray containing the posterior probabilities.
            l is the total log likelihood.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # Check dimensional alignment across all inputs
    if m.shape[0] != k or S.shape[0] != k:
        return None, None
    if m.shape[1] != d or S.shape[1] != d or S.shape[2] != d:
        return None, None

    # Check if priors sum to 1 (using np.isclose to handle float precision)
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    # Initialize the posterior probabilities matrix
    g = np.zeros((k, n))

    # Single loop over the clusters to calculate the likelihoods
    for i in range(k):
        pdf_values = pdf(X, m[i], S[i])
        if pdf_values is None:
            return None, None
        # Multiply the PDF by the cluster's prior probability
        g[i] = pi[i] * pdf_values

    # Calculate the marginal probabilities (summing across clusters)
    marginal = np.sum(g, axis=0)

    # Calculate the total log likelihood
    log_l = np.sum(np.log(marginal))

    # Normalize the likelihoods to get the true posterior probabilities (g)
    g = g / marginal

    return g, log_l

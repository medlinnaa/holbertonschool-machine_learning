#!/usr/bin/env python3
"""
Module finding the best number of clusters for a GMM using BIC.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check (inclusive).
        kmax (int): Maximum number of clusters to check (inclusive).
        iterations (int): Maximum number of iterations.
        tol (float): Tolerance of the log likelihood.
        verbose (bool): Print information to the standard output.

    Returns:
        tuple: (best_k, best_result, l, b) or 4 Nones on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return None, None, None, None

    n, d = X.shape
    log_l_all = []
    b_all = []
    results = []

    # Iterate through the requested cluster sizes
    k_range = range(kmin, kmax + 1)
    for k in k_range:
        pi, m, S, _, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None or m is None or S is None or log_l is None:
            return None, None, None, None

        # Calculate parameters count (p)
        # priors (k - 1) + means (k * d) + covariances (k * d * (d + 1) / 2)
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        # Calculate BIC value: p * ln(n) - 2 * l
        bic_val = p * np.log(n) - 2 * log_l

        log_l_all.append(log_l)
        b_all.append(bic_val)
        results.append((pi, m, S))

    b_all = np.array(b_all)
    log_l_all = np.array(log_l_all)

    # Determine the index of the lowest BIC
    best_idx = np.argmin(b_all)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_l_all, b_all

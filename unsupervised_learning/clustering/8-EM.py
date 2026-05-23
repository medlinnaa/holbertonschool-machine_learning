#!/usr/bin/env python3
"""
Module to perform the expectation maximization step of a GMM.
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization of a GMM.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations.
        tol (float): Tolerance of the log likelihood.
        verbose (bool): Boolean determining if info should be printed.

    Returns:
        tuple: (pi, m, S, g, l) or 5 Nones on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    # Step 1: Initialize the variables
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Step 2: Calculate initial expectations
    g, l = expectation(X, pi, m, S)
    if g is None or l is None:
        return None, None, None, None, None

    # Execute the EM steps up to the iterations limit
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        # Maximization step updates the parameters
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Expectation step calculates the new log likelihood
        g, l_new = expectation(X, pi, m, S)
        if g is None or l_new is None:
            return None, None, None, None, None

        # Check the early stopping condition
        if abs(l_new - l) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i + 1, round(l_new, 5)))
            l = l_new
            break

        l = l_new
    else:
        # If the loop finishes without breaking early, print the final state
        if verbose:
            print("Log Likelihood after {} iterations: {}".format(
                iterations, round(l, 5)))

    return pi, m, S, g, l

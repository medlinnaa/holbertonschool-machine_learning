#!/usr/bin/env python3
"""Module to perform a full t-SNE transformation"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        ndims (int): the new dimensional representation of X
        idims (int): the intermediate dimensional representation of X after PCA
        perplexity (float): the perplexity for the Gaussian distributions
        iterations (int): the number of iterations
        lr (float): the learning rate

    Returns:
        Y (numpy.ndarray): shape (n, ndim) containing the optimized
            low dimensional transformation of X
    """
    n, _ = X.shape

    # Step 1: Reduce initial dimensionality using PCA to speed up calculations
    X_pca = pca(X, idims)

    # Step 2: Calculate the symmetric P affinities based on the PCA reduction
    P = P_affinities(X_pca, perplexity=perplexity)

    # Step 3: Apply early exaggeration to the P affinities
    P = P * 4.0

    # Initialize the low-dimensional map Y randomly and the momentum tracker iY
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))

    # Step 4: Perform gradient descent
    for i in range(1, iterations + 1):
        # Calculate gradients (using the current Y and P)
        dY, _ = grads(Y, P)

        # Determine the momentum multiplier alpha
        alpha = 0.5 if i <= 20 else 0.8

        # Gradient descent step with momentum
        iY = alpha * iY - lr * dY
        Y = Y + iY

        # Re-center Y by subtracting its mean
        Y = Y - np.mean(Y, axis=0)

        # Stop early exaggeration after the 100th iteration
        if i == 100:
            P = P / 4.0

        # Print the cost every 100 iterations
        if i % 100 == 0:
            # Recalculate Q with the updated Y to get the accurate cost
            _, Q = grads(Y, P)
            C = cost(P, Q)
            print(f"Cost at iteration {i}: {C}")

    return Y

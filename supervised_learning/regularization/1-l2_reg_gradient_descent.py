#!/usr/bin/env python3
"""Module thagt performs gradient descent with L2 regularization """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using gradient descent with L2 regularization

    Args:
        Y: numpy.ndarray of shape (classes, m), correct labels (one-hot)
        weights: dict containing weights and biases
        cache: dict containing activations of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers
    """
    m = Y.shape[1]
    weights_copy = weights.copy()

    dZ = cache["A{}".format(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(i - 1)]
        W = weights_copy["W{}".format(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            A_prev_activation = cache["A{}".format(i - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev_activation))

        weights["W{}".format(i)] = weights["W{}".format(i)] - alpha * dW
        weights["b{}".format(i)] = weights["b{}".format(i)] - alpha * db

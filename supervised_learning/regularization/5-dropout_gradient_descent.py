#!/usr/bin/env python3
"""Gradient descent with dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights using gradient descent with dropout

    Args:
        Y: one-hot labels (classes, m)
        weights: dictionary of weights and biases
        cache: dictionary from forward propagation
        alpha: learning rate
        keep_prob: probability to keep neuron
        L: number of layers
    """
    m = Y.shape[1]

    # Output layer error
    A_L = cache['A{}'.format(L)]
    dZ = A_L - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A{}'.format(i - 1)]
        W = weights['W{}'.format(i)]

        # Gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Save weights BEFORE updating
        if i > 1:
            W_prev = weights['W{}'.format(i)]

        # Update weights
        weights['W{}'.format(i)] = W - alpha * dW
        weights['b{}'.format(i)] = weights['b{}'.format(i)] - alpha * db

        if i > 1:
            # Backprop
            dA_prev = np.matmul(W.T, dZ)

            # A¥pplydropout
            D = cache['D{}'.format(i - 1)]
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob

            # tanh derivative
            A_prev = cache['A{}'.format(i - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)

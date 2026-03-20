#!/usr/bin/env python3
"""Forward propagation with dropout"""

import numpy as np


def softmax(Z):
    """Compute softmax"""
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: input data (nx, m)
        weights: dictionary of weights and biases
        L: number of layers
        keep_prob: probability to keep a neuron

    Returns:
        cache dictionary with activations and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W{}'.format(i)]
        b = weights['b{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]

        # Linear step
        Z = np.matmul(W, A_prev) + b

        # Activation
        if i == L:
            A = softmax(Z)
        else:
            A = np.tanh(Z)

            # Dropout
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob

            cache['D{}'.format(i)] = D

        cache['A{}'.format(i)] = A

    return cache

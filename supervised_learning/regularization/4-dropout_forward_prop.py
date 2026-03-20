#!/usr/bin/env python3
"""Forward propagation with dropout"""

import numpy as np


def softmax(Z):
    """Compute softmax"""
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation with dropout"""
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W{}'.format(i)]
        b = weights['b{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            A = softmax(Z)
        else:
            A = np.tanh(Z)

            # Dropout
            D = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A = (A * D) / keep_prob

            cache['D{}'.format(i)] = D

        cache['A{}'.format(i)] = A

    return cache

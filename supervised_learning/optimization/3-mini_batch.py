#!/usr/bin/env python3
"""Module to create mini-batches"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for mini-batch gradient descent
    Args:
        X: numpy.ndarray of shape (m, nx) (input data)
        Y: numpy.ndarray of shape (m, ny) (labels)
        batch_size: number of data points in a batch
    Returns: list of mini-batches containing tuples (X_batch, Y_batch)
    """
    # 1. Shuffle the data
    X_s, Y_s = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    # 2. Slice the data into batches
    for i in range(0, m, batch_size):
        X_batch = X_s[i : i + batch_size]
        Y_batch = Y_s[i : i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches

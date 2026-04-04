#!/usr/bin/env python3
"""
Module to perform forward propagation over a pooling layer
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    # Initialize the output matrix
    pooled_out = np.zeros((m, h_out, w_out, c_prev))

    # Iterate through the output dimensions
    for i in range(h_out):
        for j in range(w_out):
            # Define the corners of the pooling window
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # Slice the input to get the pooling window
            # Shape: (m, kh, kw, c_prev)
            window = A_prev[:, v_start:v_end, h_start:h_end, :]

            # Apply the pooling operation based on the mode
            if mode == 'max':
                pooled_out[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pooled_out[:, i, j, :] = np.mean(window, axis=(1, 2))

    return pooled_out

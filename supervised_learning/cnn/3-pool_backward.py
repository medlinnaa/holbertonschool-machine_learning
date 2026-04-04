#!/usr/bin/env python3
"""
Module to perform backward propagation over a pooling layer
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer of a neural network
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize the gradient of the previous layer with zeros
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):  # Loop over examples
        for h in range(h_new):  # Loop over output height
            for w in range(w_new):  # Loop over output width
                for c in range(c_new):  # Loop over channels
                    # Define the window corners
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'max':
                        # Get the slice from the original input
                        a_prev_slice = A_prev[i, v_start:v_end,
                                        h_start:h_end, c]
                        # Create a mask where the max value was located
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # Only the max pixel gets the gradient
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += \
                            mask * dA[i, h, w, c]

                    elif mode == 'avg':
                        # The gradient is distributed equally across the window
                        avg_dA = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += \
                            np.ones(kernel_shape) * avg_dA

    return dA_prev

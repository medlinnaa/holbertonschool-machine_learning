#!/usr/bin/env python3
"""
Module to perform backward propagation over a convolutional layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation over a convolutional layer
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Initialize gradients with zeros
    dA_prev_pad = np.zeros((m, h_prev + 2 * ph, w_prev + 2 * pw, c_prev))
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad A_prev to match the dimensions used in forward prop
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    for i in range(m):  # Loop over examples
        for h in range(h_new):  # Loop over output height
            for w in range(w_new):  # Loop over output width
                for f in range(c_new):  # Loop over output channels
                    # Define the window corners in the padded input
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    # Slice the input used for this specific output pixel
                    a_slice = A_prev_pad[i, v_start:v_end, h_start:h_end, :]

                    # Update gradient for the previous layer
                    dA_prev_pad[i, v_start:v_end, h_start:h_end, :] += \
                        W[:, :, :, f] * dZ[i, h, w, f]

                    # Update gradient for the weight (kernel)
                    dW[:, :, :, f] += a_slice * dZ[i, h, w, f]

    # Remove padding from dA_prev if it was added
    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:ph+h_prev, pw:pw+w_prev, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db

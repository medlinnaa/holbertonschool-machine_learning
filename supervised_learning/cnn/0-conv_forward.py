#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Calculate output dimensions
    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    # Pad the input
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    # Initialize the output
    convolved_out = np.zeros((m, h_out, w_out, c_new))

    # Perform the convolution
    for i in range(h_out):
        for j in range(w_out):
            # Calculate the window corners
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # Slice the input image(s) for this window
            # Shape: (m, kh, kw, c_prev)
            img_slice = A_prev_pad[:, v_start:v_end, h_start:h_end, :]

            # Apply each kernel/filter
            for k in range(c_new):
                # Element-wise multiplication + Summation over height,
                # width, and input channels
                # W[:, :, :, k] has shape (kh, kw, c_prev)
                # Output shape for this step is (m,)
                Z = np.sum(img_slice * W[:, :, :, k], axis=(1, 2, 3))
                convolved_out[:, i, j, k] = Z + b[0, 0, 0, k]

    return activation(convolved_out)

#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernels: numpy.ndarray with shape (kh, kw, c, nc)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing
        the convolved images of shape (m, h_out, w_out, nc)
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Handle padding types
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        ph, pw = padding

    # Apply zero padding to height and width
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions
    out_h = ((h + (2 * ph) - kh) // sh) + 1
    out_w = ((w + (2 * pw) - kw) // sw) + 1

    # Initialize output array (m, out_h, out_w, nc)
    output = np.zeros((m, out_h, out_w, nc))

    # Perform convolution using three loops: height, width, and kernel
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw

            # Window slice across all images and channels: (m, kh, kw, c)
            image_slice = images_padded[:, h_start:h_start + kh,
                                        w_start:w_start + kw, :]

            # Loop through each kernel to create a channel in the output
            for k in range(nc):
                # Multiply slice by the k-th kernel: (kh, kw, c)
                # Sum over axes (1, 2, 3) to get shape (m,)
                output[:, i, j, k] = np.sum(image_slice * kernels[..., k],
                                            axis=(1, 2, 3))

    return output

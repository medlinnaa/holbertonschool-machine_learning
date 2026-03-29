#!/usr/bin/env python3
"""
Module to perform a convolution on images with channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # Handle padding types
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        ph, pw = padding

    # Apply zero padding only to height (axis 1) and width (axis 2)
    # We do not pad the 'm' (axis 0) or 'c' (axis 3) dimensions
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions
    out_h = ((h + (2 * ph) - kh) // sh) + 1
    out_w = ((w + (2 * pw) - kw) // sw) + 1

    # Initialize output array (m, out_h, out_w)
    # Note: The output does NOT have a channel dimension anymore
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using two loops for height and width
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw

            # Extract the window across ALL channels: (m, kh, kw, c)
            image_slice = images_padded[:, h_start:h_start + kh,
                                        w_start:w_start + kw, :]

            # Multiply slice by kernel and sum over the last 3 axes (kh, kw, c)
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return output

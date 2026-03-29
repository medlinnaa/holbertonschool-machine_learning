#!/usr/bin/env python3
"""
Module to perform pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw)
        stride: tuple of (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        A numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    # Pooling formula: floor((dim - k) / s) + 1
    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    # Initialize output array (preserving number of channels)
    output = np.zeros((m, out_h, out_w, c))

    # Perform pooling using two loops for output height and width
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw

            # Extract the window across all images and channels: (m, kh, kw, c)
            image_slice = images[:, h_start:h_start + kh,
                                 w_start:w_start + kw, :]

            # Apply the pooling operation based on the mode
            # We collapse the kh (axis 1) and kw (axis 2) dimensions
            if mode == 'max':
                output[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return output

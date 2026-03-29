#!/usr/bin/env python3
"""
Module to perform a 'same' convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images.
            m: number of images.
            h: height in pixels of the images.
            w: width in pixels of the images.
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel.
            kh: height of the kernel.
            kw: width of the kernel.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for 'same' convolution
    # Output dim = Input dim + 2p - k + 1. For Output == Input: 2p = k - 1
    # We use integer division // 2 to handle odd/even kernel sizes
    ph = kh // 2
    pw = kw // 2

    # Apply zero padding to the height and width axes (axes 1 and 2)
    # The pad_width is ((0,0), (top, bottom), (left, right))
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Initialize the output array (matches input shape h, w)
    output = np.zeros((m, h, w))

    # Perform convolution using two loops for the output dimensions
    for i in range(h):
        for j in range(w):
            # Extract slice from padded images
            # i and j here refer to the top-left corner of the window in padding
            image_slice = images_padded[:, i:i+kh, j:j+kw]
            # Multiply and sum across all images simultaneously
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output

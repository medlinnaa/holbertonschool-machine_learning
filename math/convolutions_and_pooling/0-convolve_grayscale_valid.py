#!/usr/bin/env python3
"""
Module to perform a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

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

    # Calculate dimensions of the output (Valid convolution)
    # Output height = Input height - Kernel height + 1
    # Output width = Input width - Kernel width + 1
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Initialize the output array with zeros
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using only two loops (for height and width)
    # This vectorizes the operation across all 'm' images at once
    for i in range(out_h):
        for j in range(out_w):
            # Extract the current slice from all images
            image_slice = images[:, i:i+kh, j:j+kw]
            # Element-wise multiplication and sum over the last two axes
            # result shape: (m,)
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output

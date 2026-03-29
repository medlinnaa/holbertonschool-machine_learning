#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images.
            m: number of images.
            h: height in pixels of the images.
            w: width in pixels of the images.
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel.
            kh: height of the kernel.
            kw: width of the kernel.
        padding: tuple of (ph, pw).
            ph: padding for the height of the image.
            pw: padding for the width of the image.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply the custom zero padding to the images
    # We pad the height (axis 1) and width (axis 2)
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Calculate the output dimensions based on the new padded size
    out_h = h + (2 * ph) - kh + 1
    out_w = w + (2 * pw) - kw + 1

    # Initialize the output array
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using two loops for height and width
    for i in range(out_h):
        for j in range(out_w):
            # Slice the padded images at the current window
            image_slice = images_padded[:, i:i+kh, j:j+kw]
            # Element-wise multiplication with kernel and sum over axes (1, 2)
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output

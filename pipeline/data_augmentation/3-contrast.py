#!/usr/bin/env python3
"""
Module that contains the function change_contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image

    Args:
        image: a 3D tf.Tensor containing the image to adjust
        lower: float representing the lower bound of the random contrast factor
        upper: float representing the upper bound of the random contrast factor

    Returns:
        The contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)

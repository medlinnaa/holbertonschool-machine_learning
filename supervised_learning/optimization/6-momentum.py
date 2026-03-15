#!/usr/bin/env python3
"""Module to create a Momentum optimizer in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in TF
    Args:
        alpha: the learning rate
        beta1: the momentum weight
    Returns: the optimizer
    """
    # TF's SGD optimizer implements Momentum when the momentum arg is provided
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

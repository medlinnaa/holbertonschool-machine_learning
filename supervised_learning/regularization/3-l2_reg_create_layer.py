#!/usr/bin/env python3
"""Module that creates a layer with L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer with L2 regularization

    Args:
        prev: tensor output of previous layer
        n: number of nodes in new layer
        activation: activation function
        lambtha: L2 regularization parameter

    Returns:
        output of the new layer
    """
    # Define Dense layer with L2 regularization
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg"
        ),
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )

    # Apply layer to previous output
    return layer(prev)

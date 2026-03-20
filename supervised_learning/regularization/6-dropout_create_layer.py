#!/usr/bin/env python3
"""Module that creates a layer with dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: tensor output of previous layer
        n: number of nodes
        activation: activation function
        keep_prob: probability to keep neuron
        training: boolean, whether in training mode

    Returns:
        output of the new layer
    """
    # Create Dense layer
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg"
        )
    )

    # Apply dense layer
    A = dense(prev)

    # Apply dropout ONLY during training
    if training:
        dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
        A = dropout(A, training=training)

    return A

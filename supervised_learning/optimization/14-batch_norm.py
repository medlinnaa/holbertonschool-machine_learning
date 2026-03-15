#!/usr/bin/env python3
"""
Module containing the function create_batch_norm_layer
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: the activation function to be used on the output

    Returns:
        A tensor of the activated output for the layer
    """
    # 1. Initialize the base Dense layer
    # We use VarianceScaling as requested and disable the internal bias
    # because Batch Normalization's beta parameter acts as the bias.
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=init,
                                  use_bias=False)
    z = layer(prev)

    # 2. Initialize trainable parameters gamma and beta
    # gamma: scale parameter (vector of 1s)
    # beta: offset parameter (vector of 0s)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # 3. Calculate mean and variance along the batch axis (0)
    mean, variance = tf.nn.moments(z, axes=[0])

    # 4. Apply Batch Normalization
    # Formula: gamma * (z - mean) / sqrt(variance + epsilon) + beta
    epsilon = 1e-7
    bn_output = tf.nn.batch_normalization(
        z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )

    # 5. Apply the activation function
    return activation(bn_output)

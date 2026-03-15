#!/usr/bin/env python3
"""Module to create a batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to be used on the output
    Returns: a tensor of the activated output for the layer
    """
    # 1. Initialize weights using VarianceScaling
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Create the Dense layer (Linear transformation)
    # We set use_bias=False because BN's beta parameter handles the offset
    model_layer = tf.keras.layers.Dense(units=n,
                                        kernel_initializer=init,
                                        use_bias=False)
    Z = model_layer(prev)

    # 3. Create and apply the BatchNormalization layer
    # Gamma and Beta are trainable by default (init to 1 and 0)
    bn_layer = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    Z_norm = bn_layer(Z)

    # 4. Apply the activation function last
    return activation(Z_norm)

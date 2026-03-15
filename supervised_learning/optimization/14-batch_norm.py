#!/usr/bin/env python3
"""Batch Normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow"""
    
    # 1. Setup the weight initializer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Dense Layer WITHOUT bias
    # If use_bias is True (default), it throws off the random seed sequence
    model_layer = tf.keras.layers.Dense(units=n, 
                                        kernel_initializer=init, 
                                        use_bias=False)
    
    # Apply the linear transformation
    Z = model_layer(prev)

    # 3. Apply Batch Normalization
    # Gamma and Beta are trainable by default, init to 1 and 0
    bn_layer = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    Z_norm = bn_layer(Z)

    # 4. Apply Activation last
    return activation(Z_norm)

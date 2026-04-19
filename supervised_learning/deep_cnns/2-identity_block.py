#!/usr/bin/env python3
"""Module to build an identity block for ResNet"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning
    for Image Recognition (2015)

    Args:
        A_prev: output from the previous layer
        filters: list/tuple [F11, F3, F12]
            F11: filters in first 1x1 conv
            F3: filters in 3x3 conv
            F12: filters in second 1x1 conv

    Returns: the activated output of the identity block
    """
    # Destructure the filter counts
    F11, F3, F12 = filters

    # Initialize the He Normal weight initializer with seed 0
    init = K.initializers.HeNormal(seed=0)

    # --- Component 1: First 1x1 Convolution ---
    # Purpose: Compression (Bottleneck)
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # --- Component 2: 3x3 Convolution ---
    # Purpose: Spatial feature extraction
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # --- Component 3: Second 1x1 Convolution ---
    # Purpose: Expansion (Restoring depth)
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # --- Component 4: The Skip Connection (Shortcut) ---
    # Element-wise addition of the input (A_prev) and the processed output
    add = K.layers.Add()([bn3, A_prev])

    # Final ReLU activation after the addition
    output = K.layers.Activation('relu')(add)

    return output

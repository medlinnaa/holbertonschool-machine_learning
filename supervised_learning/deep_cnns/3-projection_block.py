#!/usr/bin/env python3
"""Module to build a projection block for ResNet"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning
    for Image Recognition (2015)

    Args:
        A_prev: output from the previous layer
        filters: list/tuple [F11, F3, F12]
        s: stride of the first convolution in both main and shortcut paths

    Returns: the activated output of the projection block
    """
    # Destructure filter counts
    F11, F3, F12 = filters

    # Initialize the He Normal weight initializer with seed 0
    init = K.initializers.HeNormal(seed=0)

    # --- Main Path ---

    # First 1x1 Conv: Uses stride 's' to downsample spatially
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # 3x3 Conv: Standard processing
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Second 1x1 Conv: Expands channels to F12
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # --- Shortcut Path (The Projection) ---

    # This 1x1 Conv ensures the input (A_prev) matches the
    # dimensions and depth of the main path (bn3)
    shortcut_conv = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    shortcut_bn = K.layers.BatchNormalization(axis=3)(shortcut_conv)

    # --- Addition and Final Activation ---
    add = K.layers.Add()([bn3, shortcut_bn])
    output = K.layers.Activation('relu')(add)

    return output

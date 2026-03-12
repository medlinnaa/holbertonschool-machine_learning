#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix
    - labels: the label vector to convert
    - classes: the total number of classes
    Returns: the one-hot matrix
    """
    # Keras utility for one-hot encoding
    return K.utils.to_categorical(labels, num_classes=classes)

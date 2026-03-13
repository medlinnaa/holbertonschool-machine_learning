#!/usr/bin/env python3
"""Saves and loads only the weights of a Keras model"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a file
    """
    # Saves only the numerical parameters of the layers
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model's weights from a file
    """
    # Injects weights into an existing model architecture
    network.load_weights(filename)
    return None

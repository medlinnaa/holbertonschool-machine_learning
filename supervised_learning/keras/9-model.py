#!/usr/bin/env python3
"""Saves and loads a Keras model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file
    """
    # Saves the architecture, weights, and training configuration
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model from a file
    """
    # Reconstructs the model exactly as it was when saved
    return K.models.load_model(filename)

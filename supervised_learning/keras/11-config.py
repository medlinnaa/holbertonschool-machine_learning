#!/usr/bin/env python3
"""Saves and loads a model configuration in JSON format"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format
    """
    # Convert architecture to a JSON string
    config = network.to_json()

    # Write that string to a file
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file
    """
    # Read the JSON string from the file 
    with open(filename, 'r') as f:
        config = f.read()

    # Reconstruct the model structure from the string
    return K.models.model_from_json(config)

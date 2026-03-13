#!/usr/bin/env python3
"""Tests a Keras model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network and returns loss and accuracy
    """
    # The evaluate method runs the test data through the network
    # and compares the predictions to the true labels
    results = network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )

    return results

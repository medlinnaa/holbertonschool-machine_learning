#!/usr/bin/env python3
"""Makes a prediction using a Keras model"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network
    - network: the model to use
    - data: the input data
    - verbose: boolean for output printing
    Returns: the prediction for the data
    """
    # The predict method returns the output of the last layer (Softmax)
    return network.predict(x=data, verbose=verbose)

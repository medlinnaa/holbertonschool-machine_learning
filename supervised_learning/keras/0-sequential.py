#!/usr/bin/env python3
"""Builds a model with Keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    """
    # 1. Initialize the Sequential model
    model = K.Sequential()

    # 2. Add the first layer (needs the input shape!)
    # L2 regularization is added here via kernel_regularizer
    model.add(K.layers.Dense(
        layers[0], 
        input_shape=(nx,), 
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    ))

    # 3. Loop through the rest of the layers
    for i in range(1, len(layers)):
        # Add Dropout BEFORE the next dense layer
        # Note: Dropout takes the rate of dropping (1 - keep_prob)
        model.add(K.layers.Dropout(1 - keep_prob))

        # Add the next Dense layer
        model.add(K.layers.Dense(
            layers[i], 
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))

    return model

#!/usr/bin/env python3
"""Builds a model with Keras Functional API"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    (Functional API version)
    """
    # 1. Define the Input "placeholder"
    inputs = K.Input(shape=(nx,))

    # 2. Define the first hidden layer
    # We call the layer like a function and pass 'inputs' into it
    reg = K.regularizers.l2(lambtha)

    # This line means: Take 'inputs',
    # pass it through 'Dense', call the result 'x'
    x = K.layers.Dense(layers[0], activation=activations[0],
        kernel_regularizer=reg)(inputs)

    # 3. Loop through the rest of the layers
    for i in range(1, len(layers)):
        # Add Dropout to the previous output 'x'
        x = K.layers.Dropout(1 - keep_prob)(x)

        # Pass the result of Dropout
        # into the next Dense layer
        x = K.layers.Dense(layers[i], activation=activations[i],
            kernel_regularizer=reg)(x)

    # 4. Create the model by connecting
    # the start (inputs) to the end (x)
    model = K.Model(inputs=inputs, outputs=x)

    return model

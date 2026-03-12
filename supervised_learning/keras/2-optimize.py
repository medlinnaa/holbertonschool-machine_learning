#!/usr/bin/env python3
"""Configures a Keras model for training"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization with categorical crossentropy
    loss and accuracy metrics
    """
    # 1. Initialize the Adam Optimizer
    # alpha is the learning rate,
    # beta1 and beta2 are momentum parameters
    opt = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    # 2. Compile the model
    # This connects the optimizer and loss function to the network
    network.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return None

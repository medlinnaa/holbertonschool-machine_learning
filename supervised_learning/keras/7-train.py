#!/usr/bin/env python3
"""Trains a model with Learning Rate Decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    with learning rate decay
    """
    callbacks = []

    # 1. Early Stopping
    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    # 2. Learning Rate Decay
    if learning_rate_decay and validation_data:
        # Define the inverse time decay function
        def scheduler(epoch):
            """Calculates the learning rate for the current epoch"""
            return alpha / (1 + decay_rate * epoch)

        # Create the scheduler callback
        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)

    # Train the model
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history

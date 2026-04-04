#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 architecture using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture
    """
    # Initialize weights with He Normal and a seed of 0
    init = K.initializers.HeNormal(seed=0)

    # Layer 1: Convolutional (6 kernels, 5x5, same padding, ReLU)
    conv1 = K.layers.Conv2D(
        filters=6, kernel_size=(5, 5), padding='same',
        activation='relu', kernel_initializer=init)(X)

    # Layer 2: Max Pooling (2x2 kernels, 2x2 strides)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv1)

    # Layer 3: Convolutional (16 kernels, 5x5, valid padding, ReLU)
    conv2 = K.layers.Conv2D(
        filters=16, kernel_size=(5, 5), padding='valid',
        activation='relu', kernel_initializer=init)(pool1)

    # Layer 4: Max Pooling (2x2 kernels, 2x2 strides)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten the data so the Fully Connected layers can read it
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Fully Connected (120 nodes, ReLU)
    fc1 = K.layers.Dense(
        units=120, activation='relu', kernel_initializer=init)(flatten)

    # Layer 6: Fully Connected (84 nodes, ReLU)
    fc2 = K.layers.Dense(
        units=84, activation='relu', kernel_initializer=init)(fc1)

    # Layer 7: Output Fully Connected (10 nodes, Softmax)
    output = K.layers.Dense(
        units=10, activation='softmax', kernel_initializer=init)(fc2)

    # Create the model
    model = K.Model(inputs=X, outputs=output)

    # Compile the model with Adam and Accuracy metrics
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

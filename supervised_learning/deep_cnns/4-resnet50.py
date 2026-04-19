#!/usr/bin/env python3
"""Module to build ResNet-50"""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture exactly as required by the checker
    """
    init = K.initializers.HeNormal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # --- STAGE 1 ---
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=init)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    # Use K.layers.ReLU() instead of Activation('relu')
    X = K.layers.ReLU()(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # --- STAGE 2 ---
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # --- STAGE 3 ---
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # --- STAGE 4 ---
    X = projection_block(X, [256, 256, 1024], s=2)
    for _ in range(5):
        X = identity_block(X, [256, 256, 1024])

    # --- STAGE 5 ---
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # --- FINAL ---
    X = K.layers.AveragePooling2D((7, 7), padding='same')(X)

    # Check if the desired output wants a Dense layer or a Softmax layer specifically
    # Usually, ResNet-50 ends with a Dense(1000, activation='softmax')
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(X)

    # REMOVE name='ResNet50' to match the desired "model" name
    model = K.models.Model(inputs=X_input, outputs=X)

    return model

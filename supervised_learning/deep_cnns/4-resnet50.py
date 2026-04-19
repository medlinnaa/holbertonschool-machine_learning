#!/usr/bin/env python3
"""Module to build ResNet-50"""
from tensorflow import keras as K

# Importing your previous tasks
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture
    Returns: the keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # --- STAGE 1 ---
    # Conv1: 7x7, 64 filters, stride 2
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=init)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # MaxPool: 3x3, stride 2
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # --- STAGE 2 ---
    # 3 blocks: [1 proj, 2 identity]
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # --- STAGE 3 ---
    # 4 blocks: [1 proj, 3 identity]
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # --- STAGE 4 ---
    # 6 blocks: [1 proj, 5 identity]
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # --- STAGE 5 ---
    # 3 blocks: [1 proj, 2 identity]
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # --- FINAL AVG POOL & OUTPUT ---
    X = K.layers.AveragePooling2D((7, 7), padding='same')(X)

    # Dense layer with Softmax
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(X)

    model = K.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

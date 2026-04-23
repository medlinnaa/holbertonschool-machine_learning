#!/usr/bin/env python3
"""
Module to train a convolutional neural network for CIFAR-10
classification using transfer learning.
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model.
    X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 data
    Y: numpy.ndarray of shape (m,) containing CIFAR-10 labels
    Returns: X_p, Y_p
    """
    # X_p needs to be normalized/preprocessed as the base model expects
    # ResNet50V2 expects scaling like what's in preprocess_input
    X_p = K.applications.resnet_v2.preprocess_input(X)
    # Y_p should be one-hot encoded
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Input layer for 32x32 images
    inputs = K.Input(shape=(32, 32, 3))

    # Hint 2: Lambda layer to scale up the data to the correct size
    # ResNet50V2 expects at least 190x190 to work well (default 224x224)
    # (7, 7) scaling turns 32x32 into 224x224
    scaled_inputs = K.layers.Lambda(
        lambda x: K.backend.resize_images(x, 7, 7, "channels_last")
    )(inputs)

    # Load pre-trained ResNet50V2 model without the top classification layers
    base_model = K.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=scaled_inputs
    )

    # Hint 3: Freeze the application layers
    base_model.trainable = False

    # Add new top layers for CIFAR-10 (10 classes)
    model = K.Sequential([
        base_model,
        K.layers.Flatten(),
        K.layers.Dense(512, activation='relu'),
        K.layers.Dropout(0.3),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    # Note: Accuracy reaches 87%+ quickly because the base is pre-trained
    model.fit(
        X_train_p,
        Y_train_p,
        validation_data=(X_test_p, Y_test_p),
        batch_size=128,
        epochs=5,
        verbose=1
    )

    # Save the model
    model.save('cifar10.h5')

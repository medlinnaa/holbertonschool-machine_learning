#!/usr/bin/env python3
"""
Module for creating a Convolutional Autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): Number of filters for each convolutional layer.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
    """
    # --- Encoder ---
    encoder_inputs = keras.Input(shape=input_dims)
    encoded = encoder_inputs

    for f in filters:
        encoded = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(encoded)
        # padding='same' is crucial here to map 7x7 to 4x4 instead of 3x3
        encoded = keras.layers.MaxPooling2D(
            (2, 2), padding='same'
        )(encoded)

    encoder = keras.Model(encoder_inputs, encoded, name="encoder")

    # --- Decoder ---
    decoder_inputs = keras.Input(shape=latent_dims)
    decoded = decoder_inputs

    rev_filters = filters[::-1]

    for i in range(len(rev_filters)):
        if i == len(rev_filters) - 1:
            # The second to last convolution in the entire network
            decoded = keras.layers.Conv2D(
                rev_filters[i], (3, 3), padding='valid', activation='relu'
            )(decoded)
        else:
            decoded = keras.layers.Conv2D(
                rev_filters[i], (3, 3), padding='same', activation='relu'
            )(decoded)

        # All layers in this loop get upsampling
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    # The final convolution layer (matches input channels)
    decoder_outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid'
    )(decoded)

    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # --- Full Autoencoder ---
    auto_inputs = keras.Input(shape=input_dims)
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(auto_inputs, auto_outputs, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

#!/usr/bin/env python3
"""
Module for creating a Vanilla Autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder model.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the
            encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
    """
    # --- Encoder ---
    encoder_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder_inputs

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    latent_outputs = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(encoder_inputs, latent_outputs, name="encoder")

    # --- Decoder ---
    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoded = decoder_inputs

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # --- Full Autoencoder ---
    auto_inputs = keras.Input(shape=(input_dims,))
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(auto_inputs, auto_outputs, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

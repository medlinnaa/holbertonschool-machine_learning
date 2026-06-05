#!/usr/bin/env python3
"""
Module for creating a Variational Autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model.

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

    # Latent space parameters (no activation function)
    z_mean = keras.layers.Dense(
        latent_dims, activation=None
    )(encoded)
    z_log_var = keras.layers.Dense(
        latent_dims, activation=None
    )(encoded)

    def sampling(args):
        """Samples from the latent space normal distribution."""
        z_m, z_l_v = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(0.5 * z_l_v) * epsilon

    # Lambda layer to wrap the sampling function
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_inputs, [z, z_mean, z_log_var], name="encoder"
    )

    # --- Decoder ---
    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoded = decoder_inputs

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    decoder_outputs = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(decoded)

    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # --- Full Autoencoder ---
    auto_inputs = keras.Input(shape=(input_dims,))
    # We unpack the encoder outputs since we only need 'z' to pass to decoder
    z_out, z_m_out, z_l_v_out = encoder(auto_inputs)
    auto_outputs = decoder(z_out)
    auto = keras.Model(auto_inputs, auto_outputs, name="autoencoder")

    # --- Custom VAE Loss Function ---
    def vae_loss(y_true, y_pred):
        """Calculates the combined VAE loss."""
        # 1. Reconstruction Loss
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true, y_pred)
        # Keras BCE calculates the mean; we multiply to get the sum
        reconstruction_loss *= input_dims

        # 2. Kullback-Leibler (KL) Divergence
        kl_loss = (1 + z_l_v_out - keras.backend.square(z_m_out) -
                   keras.backend.exp(z_l_v_out))
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return reconstruction_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto

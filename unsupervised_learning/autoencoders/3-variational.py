#!/usr/bin/env python3
"""Module for creating a variational autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): dimensions of the model input.
        hidden_layers (list): number of nodes for each hidden layer in
            the encoder, respectively (reversed for the decoder).
        latent_dims (int): dimensions of the latent space representation.

    Returns:
        encoder, decoder, auto:
            encoder: the encoder model, outputting the latent
                representation, the mean, and the log variance.
            decoder: the decoder model.
            auto: the full autoencoder model, compiled with adam
                optimization and binary cross-entropy loss.
    """
    # --- Encoder ---
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick: sample z from N(mean, var)."""
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,))([mean, log_var])
    encoder = keras.Model(encoder_inputs, [z, mean, log_var])

    # --- Decoder ---
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims,
                                         activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # --- Full autoencoder ---
    z_out, mean_out, log_var_out = encoder(encoder_inputs)
    outputs = decoder(z_out)
    auto = keras.Model(encoder_inputs, outputs)

    # KL divergence term registered on the model's real forward tensors.
    kl = 1 + log_var_out - keras.backend.square(mean_out)
    kl = kl - keras.backend.exp(log_var_out)
    kl = -0.5 * keras.backend.sum(kl, axis=1)
    auto.add_loss(keras.backend.mean(kl))

    def reconstruction_loss(y_true, y_pred):
        """Binary cross-entropy summed over the input dimensions."""
        bce = keras.backend.binary_crossentropy(y_true, y_pred)
        return keras.backend.sum(bce, axis=1)

    auto.compile(optimizer='adam', loss=reconstruction_loss)

    return encoder, decoder, auto

#!/usr/bin/env python3
"""
Variational Autoencoder module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
            layer in the encoder, respectively. The hidden layers should
            be reversed for the decoder.
        latent_dims: integer containing the dimensions of the latent space
            representation

    Returns:
        encoder: the encoder model, which outputs the latent representation,
            the mean, and the log variance, respectively
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """
        Samples a vector from the latent space using the reparameterization trick.
        """
        z_m, z_l_v = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(0.5 * z_l_v) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var])

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs)

    # -------------------------------------------------------------------------
    # Full Autoencoder
    # -------------------------------------------------------------------------
    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, auto_outputs)

    def vae_loss(y_true, y_pred):
        """
        Calculates the VAE loss combining Binary Cross-Entropy and KL divergence.
        """
        # Reconstruction loss scaled by input_dims to act as sum over features
        reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        reconstruction_loss *= input_dims

        # Kullback-Leibler divergence
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return reconstruction_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto

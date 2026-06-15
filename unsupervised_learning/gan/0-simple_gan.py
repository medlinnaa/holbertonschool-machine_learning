#!/usr/bin/env python3
"""
Simple GAN module.
Contains the Simple_GAN class used to train basic generator
and discriminator models.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """
    A simple Generative Adversarial Network (GAN) model.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initializes the Simple_GAN instance.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))

        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)

        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generates a sample of fake examples using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Retrieves a random batch of real examples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Executes a single training step for the GAN.
        Updates the discriminator disc_iter times, then updates the
        generator once.
        """
        # Update the discriminator self.disc_iter times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                # Compute loss on real and fake samples
                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

            # apply gradient descent once to the discriminator
            discr_gradients = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables)
            )

        # compute the loss for the generator
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)

            # compute the loss gen_loss of the generator on this sample
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        # apply gradient descent to the generator
        gen_gradients = tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

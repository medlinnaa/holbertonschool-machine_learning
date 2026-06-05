# Autoencoders

> Implementation of various Autoencoder architectures using TensorFlow/Keras.


## 📖Description

This project explores the domain of representation learning by building several types of autoencoders from scratch. Autoencoders are unsupervised neural networks designed to compress input data into a lower-dimensional representation (the bottleneck or latent space) and subsequently reconstruct the original data as accurately as possible.

Rather than focusing on a single model, this repository provides a comprehensive pipeline that spans from simple feed-forward designs to complex generative models. By progressing through different architectures, the project demonstrates how altering the network's structure, applying regularization techniques, and shifting to probabilistic frameworks can dictate what the model learns and how effectively it compresses and reconstructs spatial or structural information.


## 🧠Concepts

* **Bottleneck Architecture:** Constructing a symmetrical encoder-decoder network that forces data through a heavily constrained dimension, ensuring the model learns meaningful representations rather than simply copying the input.
* **Sparsity Constraints:** Applying L1 regularization directly to the activity of the latent layer to create a Sparse Autoencoder. This penalizes the model for overusing neurons, forcing it to learn distinct, robust features.
* **Spatial Compression:** Utilizing `Conv2D`, `MaxPooling2D`, and `UpSampling2D` layers in Convolutional Autoencoders to handle image data. This allows the model to learn localized feature maps and maintain spatial hierarchies, which standard dense networks lose.
* **Probabilistic Latent Spaces:** Transitioning from deterministic mappings to generative models via Variational Autoencoders (VAEs). Instead of mapping inputs to a fixed vector, the encoder outputs the parameters (mean and log-variance) of a probability distribution.
* **The Reparameterization Trick:** Enabling backpropagation through a stochastic node in a VAE by randomly sampling an $\epsilon$ from a standard normal distribution, keeping the gradient flow intact.
* **Custom Loss Formulation:** Balancing reconstruction loss (Binary Cross-Entropy) with the Kullback-Leibler (KL) Divergence in VAEs, which acts as a regularizer to ensure the learned latent distributions closely resemble a standard normal distribution.


## ⚙️Requirements

* Python 3.9
* TensorFlow 2.15
* NumPy 1.25.2
* Ubuntu 20.04

# Generative Adversarial Networks (GANs)

> Implementation of various Generative Adversarial Networks, including Simple GANs, WGANs, and WGAN-GPs, using TensorFlow/Keras to generate point distributions and artificial faces.


## 📖Description

This project implements a complete progression of Generative Adversarial Network (GAN) architectures from scratch using TensorFlow. 

By leveraging deep learning, the project explores the adversarial training process where two networks—a Generator and a Discriminator—compete against each other. The project begins by generating simple 2D and 5D point cloud distributions to understand the basic mechanics. It then addresses common GAN training instabilities (such as vanishing gradients) by implementing Wasserstein GANs with weight clipping, and subsequently upgrading to the more robust Gradient Penalty method. Finally, the project culminates in building a deep convolutional GAN to generate 16x16 human faces, demonstrating its superiority over traditional Principal Component Analysis (PCA) by successfully modeling complex feature dependencies.


## 🧠Concepts

* **Minimax Games:** Implementing the dual-training loop where a Generator learns to fool a Discriminator, while the Discriminator learns to distinguish between real and fake data.
* **Wasserstein Distance:** Modifying the standard loss functions to use the Earth Mover's Distance, improving gradient flow and overall training stability in higher dimensions.
* **Lipschitz Constraints:** Enforcing mathematical constraints on the Discriminator using either rudimentary Weight Clipping (`tf.clip_by_value`) or the more advanced Gradient Penalty method.
* **Deep Convolutional GANs:** Utilizing `UpSampling2D`, `Conv2D`, and `BatchNormalization` layers to build architectures capable of processing and generating spatial image data.
* **Latent Space Exploration:** Sampling from normal, uniform, or spherical distributions and observing how the generator maps this latent space into complex, high-dimensional target distributions.
* **GANs vs. PCA:** Analyzing why standard PCA fails to generate realistic faces (due to the lack of independence in principal coordinates) and how GANs overcome this by capturing the true dependencies between features.


## ⚙️Requirements

* Python 3.9
* TensorFlow 2.15.0
* NumPy 1.25.2
* Ubuntu 20.04 LTS

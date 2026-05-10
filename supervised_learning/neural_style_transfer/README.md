# Neural Style Transfer

> Implementation of Neural Style Transfer using TensorFlow/Keras and the VGG19 model.

---

## Description

This project implements a complete Neural Style Transfer (NST) pipeline from scratch using TensorFlow. 

By leveraging deep learning, the algorithm takes two images—a content reference (like a photograph) and a style reference (like a painting)—and fuses them together. Instead of training a model to classify images, this project uses a pre-trained network to act as an artistic evaluator. It iteratively updates the pixels of a generated image until it successfully mimics the structural layout of the photograph while adopting the color palette and brushstrokes of the artwork.

---

## Concepts

* Feature Extraction: Utilizing the intermediate layers of a pre-trained VGG19 network to isolate high-level structure from low-level textures.
* Gram Matrices: Calculating the dot products of feature maps to capture artistic style (color correlations and patterns) independently of spatial location.
* Loss Functions: Formulating a custom objective function that balances content cost (retaining structure), style cost (applying textures), and variational cost (smoothing pixel noise).
* Pixel Optimization: Performing gradient descent with Adam optimization directly on the image tensor itself, rather than updating network weights.

---

## Requirements

* Python 3.9
* TensorFlow 2.15
* NumPy 1.25.2
* Ubuntu 20.04

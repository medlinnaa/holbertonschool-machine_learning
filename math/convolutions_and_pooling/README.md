# Convolutions and Pooling

> Implementation of convolution and pooling operations for grayscale and multi-channel images using NumPy.


## 📖 Description

This project implements the core building blocks of **Convolutional Neural Networks (CNNs)** from scratch using NumPy.

It covers:

* Grayscale convolution (valid, same, and custom padding)
* Strided convolution
* Multi-channel convolution
* Multi-kernel convolution
* Pooling operations (max and average)

These operations are fundamental for feature extraction and dimensionality reduction in deep learning models.


## 🧠 Concepts

* Convolution: sliding kernel over input to extract features
* Padding:

  * **Valid** → no padding
  * **Same** → output size matches input
  * **Custom** → user-defined padding
* Stride: controls step size of kernel movement
* Multi-channel convolution: processes RGB-like inputs
* Multiple kernels: produces multiple feature maps
* Pooling:

  * Max pooling → selects strongest activation
  * Average pooling → smooths features and reduces noise


## ⚙️ Requirements

* Python 3.9
* NumPy 1.25.2
* Ubuntu 20.04

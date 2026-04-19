# Regularization

> Implementation of regularization techniques including L2 regularization, dropout, and early stopping for neural networks.


## 📖 Description

This project implements key **regularization techniques** used to prevent overfitting in deep learning models.

It includes:

* L2 regularization (cost + gradient updates)
* L2 regularization in Keras models and layers
* Dropout forward propagation
* Dropout gradient descent
* Dropout layers in TensorFlow
* Early stopping mechanism

These techniques improve model generalization and training stability.


## 🧠 Concepts

* **L2 Regularization**

  * Adds penalty on large weights: `λ * ||W||²`
  * Reduces overfitting by discouraging complex models

* **Gradient Descent with L2**

  * Updates weights while penalizing large values

* **Dropout**

  * Randomly disables neurons during training
  * Prevents co-adaptation of features

* **Early Stopping**

  * Stops training when validation loss stops improving
  * Prevents unnecessary training and overfitting


## ⚙️ Requirements

* Python 3.9
* NumPy 1.25.2
* TensorFlow 2.15
* Ubuntu 20.04

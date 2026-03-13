# Keras Supervised Learning

## Project Overview
This repository contains a comprehensive implementation of neural network development using the TensorFlow 2 and Keras platforms. I developed this project to demonstrate a complete machine learning workflow—moving from raw data preprocessing to deploying a trained model for inference. The project focuses on building modular, reusable code for constructing deep learning architectures while maintaining high standards for code quality and mathematical accuracy.

## Technical Architecture
The core of this project is built upon the Sequential and Functional APIs. I implemented various architectural patterns to understand the flow of tensors through a network, utilizing the Functional API specifically to handle more complex input/output structures. 

> To improve model generalization and prevent overfitting, the layers incorporate L2 Regularization and Dropout techniques.



## Optimization and Training Strategy
I configured the Adam optimizer to handle weight updates, allowing for granular control over the learning rate (alpha) and momentum parameters (beta1 and beta2). The training process utilizes mini-batch gradient descent and incorporates advanced Keras callbacks to ensure efficiency:

* **Early Stopping:** Monitoring validation loss to terminate training once the model stops improving, thus preventing over-specialization on training data.
* **Learning Rate Decay:** Implementing an inverse time decay schedule that reduces the learning rate in a stepwise fashion after each epoch to allow for better convergence.
* **Model Checkpointing:** Implementing logic to preserve only the iteration of the model that achieves the lowest validation loss during the training cycle.



## Data Preprocessing and Evaluation
To prepare the data for the networks, I implemented custom one-hot encoding for label vectors to satisfy categorical crossentropy loss requirements. The project concludes with a testing module that evaluates the trained model on unseen data, providing final metrics for loss and accuracy, followed by an inference script to generate Softmax probability distributions for new inputs.


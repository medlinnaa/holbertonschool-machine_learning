# Transfer Learning

> Implementation of transfer learning using a pretrained ResNet50V2 model for CIFAR-10 classification.


## 📖 Description

This project applies **transfer learning** to image classification using the **CIFAR-10 dataset**.

A pretrained **ResNet50V2** model (trained on **ImageNet dataset**) is reused as a feature extractor.
Custom classification layers are added on top to adapt the model to the CIFAR-10 task.

The base model is frozen, and only the newly added layers are trained.


## 🧠 Concepts

* Transfer learning: reuse of pretrained models for new tasks
* Feature extraction: using frozen pretrained layers
* Fine-tuning: optional unfreezing of layers for adaptation
* Input preprocessing for pretrained networks
* Image resizing to match model requirements
* One-hot encoding for classification


## ⚙️ Requirements

* Python 3.9
* TensorFlow 2.15
* NumPy 1.25.2
* Ubuntu 20.04

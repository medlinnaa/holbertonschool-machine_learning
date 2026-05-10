# Object Detection

> Implementation of the YOLOv3 object detection algorithm pipeline using TensorFlow/Keras and OpenCV.


## 📖 Description

This project implements a complete pre-processing and post-processing pipeline for the YOLO (You Only Look Once) v3 object detection model from scratch using Python. 

By leveraging computer vision and deep learning libraries, the algorithm takes raw images from a directory, formats them to the network's strict specifications, and processes the raw neural network predictions. Instead of training the Darknet model, this project focuses on the complex mathematics required to decode the model's output tensors. It transforms raw log-space offsets into physical bounding boxes, filters out low-confidence background noise, and resolves duplicate detections to output clean, accurate bounding boxes around detected objects.


## 🧠 Concepts

* Output Decoding: Applying Sigmoid and Exponential functions to transform raw network predictions (logits and offsets) into relative spatial coordinates and bounding box dimensions.
* Box Filtering: Calculating a final prediction score by combining objectness confidence with individual class probabilities to discard weak predictions based on a defined threshold.
* Non-Max Suppression (NMS): Utilizing the Intersection over Union (IoU) metric to identify and eliminate redundant, overlapping bounding boxes that predict the same object.
* Image Preprocessing: Using OpenCV to read file directories, correct color space mismatching (BGR to RGB), apply inter-cubic interpolation for resizing, and normalize pixel arrays for model inference.


## ⚙️ Requirements

* Python 3.9
* TensorFlow 2.15
* NumPy 1.25.2
* Ubuntu 20.04

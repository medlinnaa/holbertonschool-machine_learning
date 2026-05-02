#!/usr/bin/env python3
"""
Contains the Yolo class to perform object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names used
                          for the Darknet model can be found
            class_t: float representing the box score threshold
            nms_t: float representing the IOU threshold for NMS
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        # Load the model using Keras
        self.model = K.models.load_model(model_path, compile=False)

        # Read class names from the file and store as a list
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

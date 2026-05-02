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
        """
        self.model = K.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs into boundary boxes, confidences,
        and class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, n_anchors, _ = output.shape

            # 1. Extract raw data
            # t_x, t_y, t_w, t_h
            raw_boxes = output[..., :4]
            # Box confidence (Objectness score)
            conf = output[..., 4:5]
            # Class probabilities
            cls_probs = output[..., 5:]

            # 2. Apply Sigmoid to center coordinates and confidence
            # This ensures values are between 0 and 1
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            t_x = sigmoid(raw_boxes[..., 0])
            t_y = sigmoid(raw_boxes[..., 1])
            t_w = raw_boxes[..., 2]
            t_h = raw_boxes[..., 3]

            box_confidences.append(sigmoid(conf))
            box_class_probs.append(sigmoid(cls_probs))

            # 3. Create Grid System
            # We need to know which cell (cx, cy) we are in
            cx = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
            cy = np.tile(np.arange(grid_h), (grid_w, 1)).T

            cx = cx[:, :, np.newaxis]
            cy = cy[:, :, np.newaxis]

            # 4. Transform to Relative Coordinates
            # Formulas: bx = sigmoid(tx) + cx / grid_width
            bx = (t_x + cx) / grid_w
            by = (t_y + cy) / grid_h

            # Anchor transformation: bw = anchor_w * exp(tw) / model_input_w
            # self.model.input.shape[1] is usually 416
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            bw = (self.anchors[i, :, 0] * np.exp(t_w)) / input_w
            bh = (self.anchors[i, :, 1] * np.exp(t_h)) / input_h

            # 5. Convert to Corner Coordinates (x1, y1, x2, y2)
            # Scaled to the original image size
            img_h, img_w = image_size

            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            processed_boxes = np.zeros(raw_boxes.shape)
            processed_boxes[..., 0] = x1
            processed_boxes[..., 1] = y1
            processed_boxes[..., 2] = x2
            processed_boxes[..., 3] = y2

            boxes.append(processed_boxes)

        return (boxes, box_confidences, box_class_probs)

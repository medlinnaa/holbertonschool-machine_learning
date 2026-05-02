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

            raw_boxes = output[..., :4]
            conf = output[..., 4:5]
            cls_probs = output[..., 5:]

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            t_x = sigmoid(raw_boxes[..., 0])
            t_y = sigmoid(raw_boxes[..., 1])
            t_w = raw_boxes[..., 2]
            t_h = raw_boxes[..., 3]

            box_confidences.append(sigmoid(conf))
            box_class_probs.append(sigmoid(cls_probs))

            cx = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
            cy = np.tile(np.arange(grid_h), (grid_w, 1)).T
            cx = cx[:, :, np.newaxis]
            cy = cy[:, :, np.newaxis]

            bx = (t_x + cx) / grid_w
            by = (t_y + cy) / grid_h

            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            bw = (self.anchors[i, :, 0] * np.exp(t_w)) / input_w
            bh = (self.anchors[i, :, 1] * np.exp(t_h)) / input_h

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on object confidence and class probabilities
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Step 1: Calculate the score for each box
            # Score = Box Confidence * Max Class Probability
            scores = box_confidences[i] * box_class_probs[i]

            # Step 2: Find the class index with the highest score
            classes = np.argmax(scores, axis=-1)
            # Get the actual score value of that class
            class_scores = np.max(scores, axis=-1)

            # Step 3: Create a mask of boxes that exceed our threshold
            mask = class_scores >= self.class_t

            # Step 4: Use the mask to select only the "good" boxes
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        # Concatenate lists from all 3 scales into single numpy arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

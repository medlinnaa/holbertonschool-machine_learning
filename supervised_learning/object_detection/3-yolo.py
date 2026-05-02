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
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        return (np.concatenate(filtered_boxes, axis=0),
                np.concatenate(box_classes, axis=0),
                np.concatenate(box_scores, axis=0))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max Suppression to the filtered bounding boxes
        """
        nms_boxes = []
        nms_classes = []
        nms_scores = []

        # We must process one class at a time
        for cls in np.unique(box_classes):
            # Get indices of boxes belonging to this specific class
            indices = np.where(box_classes == cls)[0]

            # Extract the boxes and scores for this class
            cls_boxes = filtered_boxes[indices]
            cls_scores = box_scores[indices]

            # Coordinates for IOU calculation
            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)

            # Sort indices by score (descending)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                # Keep the best box
                i = order[0]
                keep.append(i)

                if order.size == 1:
                    break

                # Find intersection with the rest
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                inter = w * h

                # Intersection over Union (IOU) formula
                iou = inter / (areas[i] + areas[order[1:]] - inter)

                # Keep indices where overlap is less than the threshold
                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            # Store the surviving boxes for this class
            nms_boxes.append(cls_boxes[keep])
            nms_classes.append(np.full(len(keep), cls))
            nms_scores.append(cls_scores[keep])

        # Final assembly: concatenate all boxes and return
        box_predictions = np.concatenate(nms_boxes, axis=0)
        predicted_box_classes = np.concatenate(nms_classes, axis=0)
        predicted_box_scores = np.concatenate(nms_scores, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

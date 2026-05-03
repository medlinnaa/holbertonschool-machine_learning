#!/usr/bin/env python3
"""
Contains the Yolo class to perform object detection
"""
import tensorflow.keras as K
import numpy as np
import cv2
import os


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
        and class probabilities
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, n_anchors, _ = output.shape

            # Extract box confidence and class probabilities using sigmoid
            conf = 1 / (1 + np.exp(-output[..., 4:5]))
            cls_probs = 1 / (1 + np.exp(-output[..., 5:]))

            # Decode bounding box coordinates
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Create grid for center offsets
            cx = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
            cy = np.tile(np.arange(grid_h), (grid_w, 1)).T
            cx = cx[:, :, np.newaxis]
            cy = cy[:, :, np.newaxis]

            # Sigmoid center coordinates + grid offset, normalized by grid size
            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_h

            # Width and Height using anchors and exponential
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]
            bw = (self.anchors[i, :, 0] * np.exp(t_w)) / input_w
            bh = (self.anchors[i, :, 1] * np.exp(t_h)) / input_h

            # Convert to (x1, y1, x2, y2) and scale to original image size
            img_h, img_w = image_size
            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            p_boxes = np.zeros(output[..., :4].shape)
            p_boxes[..., 0] = x1
            p_boxes[..., 1] = y1
            p_boxes[..., 2] = x2
            p_boxes[..., 3] = y2

            boxes.append(p_boxes)
            box_confidences.append(conf)
            box_class_probs.append(cls_probs)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on object confidence and class probabilities
        """
        f_boxes, f_classes, f_scores = [], [], []

        for i in range(len(boxes)):
            # Calculate combined score
            scores = box_confidences[i] * box_class_probs[i]

            # Find the best class and its score for each box
            box_classes = np.argmax(scores, axis=-1)
            box_scores = np.max(scores, axis=-1)

            # Filter using the threshold
            mask = box_scores >= self.class_t

            f_boxes.append(boxes[i][mask])
            f_classes.append(box_classes[mask])
            f_scores.append(box_scores[mask])

        return (np.concatenate(f_boxes),
                np.concatenate(f_classes),
                np.concatenate(f_scores))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max Suppression to the filtered bounding boxes
        """
        keep_boxes, keep_classes, keep_scores = [], [], []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            x1, y1 = cls_boxes[:, 0], cls_boxes[:, 1]
            x2, y2 = cls_boxes[:, 2], cls_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                if order.size == 1:
                    break

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            keep_boxes.append(cls_boxes[keep])
            keep_classes.append(np.full(len(keep), cls))
            keep_scores.append(cls_scores[keep])

        return (np.concatenate(keep_boxes),
                np.concatenate(keep_classes),
                np.concatenate(keep_scores))

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder path
        """
        images = []
        image_paths = []
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)
        return (images, image_paths)

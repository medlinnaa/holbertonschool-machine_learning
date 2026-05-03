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
        """ Class constructor """
        self.model = K.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ Decodes raw Darknet outputs into boundary boxes """
        boxes, confidences, probs = [], [], []
        ih, iw = image_size
        mw, mh = self.model.input.shape[1:3]

        for i, out in enumerate(outputs):
            gh, gw = out.shape[:2]
            confidences.append(1 / (1 + np.exp(-out[..., 4:5])))
            probs.append(1 / (1 + np.exp(-out[..., 5:])))

            c = np.indices((gw, gh)).transpose(2, 1, 0)
            bxby = (1 / (1 + np.exp(-out[..., :2])) + c) / [gw, gh]
            bwbh = (self.anchors[i] * np.exp(out[..., 2:4])) / [mw, mh]

            p_box = np.zeros(out[..., :4].shape)
            p_box[..., :2] = (bxby - bwbh / 2) * [iw, ih]
            p_box[..., 2:] = (bxby + bwbh / 2) * [iw, ih]
            boxes.append(p_box)

        return boxes, confidences, probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ Filters boxes by class score threshold """
        b, c, s = [], [], []
        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t
            b.append(boxes[i][mask]), c.append(classes[mask])
            s.append(class_scores[mask])
        return np.concatenate(b), np.concatenate(c), np.concatenate(s)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Removes overlapping redundant boxes """
        kb, kc, ks = [], [], []
        for cl in np.unique(box_classes):
            idx = np.where(box_classes == cl)[0]
            b, s = filtered_boxes[idx], box_scores[idx]
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            areas, order = (x2 - x1 + 1) * (y2 - y1 + 1), s.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), \
                    np.maximum(y1[i], y1[order[1:]])
                xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), \
                    np.minimum(y2[i], y2[order[1:]])
                inter = np.maximum(0, xx2 - xx1 + 1) * \
                    np.maximum(0, yy2 - yy1 + 1)
                iou = inter / (areas[i] + areas[order[1:]] - inter)
                order = order[np.where(iou <= self.nms_t)[0] + 1]
            kb.append(b[keep]), kc.append(np.full(len(keep), cl)), \
                ks.append(s[keep])
        return np.concatenate(kb), np.concatenate(kc), np.concatenate(ks)

    @staticmethod
    def load_images(folder_path):
        """ Loads all images from a folder path """
        imgs, paths = [], []
        for f in os.listdir(folder_path):
            p = os.path.join(folder_path, f)
            img = cv2.imread(p)
            if img is not None:
                imgs.append(img), paths.append(p)
        return imgs, paths

    def preprocess_images(self, images):
        """
        Prepares images for the Yolo model
        """
        # Get expected input shape from the model and cast to int
        input_h = int(self.model.input.shape[1])
        input_w = int(self.model.input.shape[2])

        pimages = []
        image_shapes = []

        for img in images:
            # 1. Store original shape (height, width)
            image_shapes.append(img.shape[:2])

            # 2. Resize using Inter-Cubic Interpolation
            # Note: cv2.resize takes (width, height)
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            # 3. Rescale pixel values to [0, 1]
            rescaled = resized / 255.0
            pimages.append(rescaled)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and box scores
        """
        for i in range(len(boxes)):
            # Cast coordinates to int for OpenCV drawing
            x1, y1, x2, y2 = boxes[i].astype(int)

            # Draw the boundary box - Blue: (255, 0, 0), Thickness: 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Prepare text: "class_name score" (score rounded to 2 decimals)
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = "{} {:.2f}".format(class_name, score)

            # Draw text - Red: (0, 0, 255), 5px above top-left
            # FONT_HERSHEY_SIMPLEX, scale 0.5, thickness 1, LINE_AA
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        1, cv2.LINE_AA)

        # Show the image window
        cv2.imshow(file_name, image)

        # Wait for key press
        key = cv2.waitKey(0)

        # Save image if 's' key is pressed
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            path = os.path.join('detections', file_name)
            cv2.imwrite(path, image)

        # Clean up windows
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Runs the full prediction pipeline on a folder of images
        """
        # 1. Load all images from the specified folder
        images, image_paths = self.load_images(folder_path)

        # 2. Preprocess all images for the model
        pimages, image_shapes = self.preprocess_images(images)

        # 3. Perform the actual prediction using the Keras model
        # The model expects a 4D tensor (ni, input_h, input_w, 3)
        outputs = self.model.predict(pimages)

        predictions = []

        # 4. Process each image's output individually
        for i in range(len(images)):
            # Extract the raw output for this specific image
            # Yolo v3 has 3 output layers for different scales
            image_outputs = [output[i] for output in outputs]

            # Decode the raw outputs into boxes, confidences, and probabilities
            boxes, confidences, probs = self.process_outputs(
                image_outputs, image_shapes[i])

            # Filter boxes based on the class threshold
            f_boxes, f_classes, f_scores = self.filter_boxes(
                boxes, confidences, probs)

            # Apply Non-Max Suppression to remove redundant overlapping boxes
            nms_boxes, nms_classes, nms_scores = self.non_max_suppression(
                f_boxes, f_classes, f_scores)

            # Append the final prediction tuple for this image
            predictions.append((nms_boxes, nms_classes, nms_scores))

            # 5. Display the boxes on the original image
            # Filename for window title should be the base name (e.g., 'dog.jpg')
            file_name = os.path.basename(image_paths[i])
            self.show_boxes(images[i], nms_boxes, nms_classes,
                            nms_scores, file_name)

        return predictions, image_paths

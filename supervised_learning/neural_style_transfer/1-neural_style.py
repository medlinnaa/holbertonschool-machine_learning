#!/usr/bin/env python3
"""
Module for Neural Style Transfer (NST).
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for Neural Style Transfer.
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class instance.

        Args:
            style_image (numpy.ndarray): The style reference image.
            content_image (numpy.ndarray): The content reference image.
            alpha (float): The weight for content cost.
            beta (float): The weight for style cost.
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if type(alpha) not in (int, float) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if type(beta) not in (int, float) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (numpy.ndarray): The image to be scaled.

        Returns:
            tf.Tensor: The scaled image.
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]
        max_dim = 512
        scale = max_dim / max(h, w)
        new_shape = (int(h * scale), int(w * scale))

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(
            image, new_shape, method=tf.image.ResizeMethod.BICUBIC)
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost.
        Replaces MaxPooling2D with AveragePooling2D
        and targets specific layers.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        # Reconstruct the model to replace max pooling with average pooling
        x = vgg.input
        model_outputs = {}
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)
            model_outputs[layer.name] = x

        # Gather the specified outputs in order
        outputs = [model_outputs[name] for name in
                   self.style_layers + [self.content_layer]]

        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        self.model.trainable = False

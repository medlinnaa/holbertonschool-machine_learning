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
        """Initializes the NST class instance."""
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            err = "style_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(err)

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            err = "content_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(err)

        if type(alpha) not in (int, float) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if type(beta) not in (int, float) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales an image: pixels in [0, 1] and max side 512 pixels."""
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            err = "image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(err)

        h, w = image.shape[:2]
        scale = 512 / max(h, w)
        new_shape = (int(h * scale), int(w * scale))

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(
            image, new_shape, method=tf.image.ResizeMethod.BICUBIC)
        image = image / 255.0

        return tf.clip_by_value(image, 0.0, 1.0)

    def load_model(self):
        """Creates the VGG19 model used to calculate cost."""
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        x = vgg.input
        model_outputs = {}
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding, name=layer.name)(x)
            else:
                x = layer(x)
            model_outputs[layer.name] = x

        outputs = [
            model_outputs[name] for name in
            self.style_layers + [self.content_layer]
        ]

        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a given tensor."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bhwc,bhwd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations

    def generate_features(self):
        """Extracts the features used to calculate neural style cost."""
        # VGG19 requires inputs scaled 0-255 and specific preprocessing
        # like reversing RGB to BGR and subtracting ImageNet mean values.
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0)

        # Pass the images through the loaded model
        style_outputs = self.model(preprocessed_style)
        content_outputs = self.model(preprocessed_content)

        # Style outputs are all except the last one
        self.gram_style_features = [
            self.gram_matrix(layer) for layer in style_outputs[:-1]
        ]

        # Content feature is the final output layer
        self.content_feature = content_outputs[-1]

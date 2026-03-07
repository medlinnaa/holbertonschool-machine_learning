#!/usr/bin/env python3
"""Defines a deep neural network"""
import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network
        nx: number of input features
        layers: list of nodes per layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            # First, check if layer nodes are valid
            if not isinstance(layers[l-1], int) or layers[l-1] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine input size of the current layer
            # Layer 1 uses nx; subsequent layers use the size of previous layer
            prev_size = nx if l == 1 else layers[l-2]

            # W = randn * sqrt(2 / prev_layer_size)
            self.weights['W' + str(l)] = np.random.randn(
                layers[l-1], prev_size) * np.sqrt(2 / prev_size)

            # Biases initialized to zeros
            self.weights['b' + str(l)] = np.zeros((layers[l-1], 1))

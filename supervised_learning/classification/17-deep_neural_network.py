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
        layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Use 'layer' to avoid pycodestyle E741 error
        for layer in range(1, self.__L + 1):
            if not isinstance(layers[layer - 1], int) or layers[layer - 1] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine input size of current layer
            prev_size = nx if layer == 1 else layers[layer - 2]

            self.__weights['W' + str(layer)] = np.random.randn(
                layers[layer - 1], prev_size) * np.sqrt(2 / prev_size)

            # Bias initialized to zeros
            self.__weights['b' + str(layer)] = np.zeros((layers[layer - 1], 1))

    @property
    def L(self):
        """Getter __L"""
        return self.__L

    @property
    def cache(self):
        """Getter __cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter __weights"""
        return self.__weights

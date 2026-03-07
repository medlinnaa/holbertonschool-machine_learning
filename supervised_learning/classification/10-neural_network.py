#!/usr/bin/env python3
"""Defines a neural network with one hidden layer"""
import numpy as np


class NeuralNetwork:
    """
    Class that defines a neural network performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network
        nx: number of input features
        nodes: number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private attributes of hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Private attributes of output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network
        X: numpy.ndarray with shape (nx, m) containing input data
        Returns: private attributes __A1 and __A2, respectively
        """
        # 1. Hidden Layer calculation
        # Z1 = W1 * X + b1
        z1 = np.matmul(self.__W1, X) + self.__b1
        # A1 = Sigmoid activation of z1
        self.__A1 = 1 / (1 + np.exp(-z1))

        # 2. Output Layer calculation
        # Z2 = W2 * A1 + b2
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # A2 = Sigmoid activation of z2 (final prediction)
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

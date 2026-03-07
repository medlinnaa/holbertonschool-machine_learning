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
        # Hidden Layer calculation
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        # Output Layer calculation
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates model cost using logistic regression
        Y: numpy.ndarray with shape (1, m) containing correct labels
        A: numpy.ndarray with shape (1, m) containing activated outputs
        Returns: the cost
        """
        m = Y.shape[1]
        # Use 1.0000001 - A to avoid division by zero
        inner_log = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(inner_log)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network predictions
        X: numpy.ndarray with shape (nx, m) containing input data
        Y: numpy.ndarray with shape (1, m) containing correct labels
        Returns: prediction and cost
        """
        # Get activated outputs from the layers
        _, a2 = self.forward_prop(X)
        # Calculate the error of the model
        cost = self.cost(Y, a2)
        # Convert output probabilities to binary labels (0 or 1)
        prediction = np.where(a2 >= 0.5, 1, 0)

        return prediction, cost

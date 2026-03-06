#!/usr/bin/env python3
"""Defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initializes the neuron
        nx: number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter __W"""
        return self.__W

    @property
    def b(self):
        """Getter __b"""
        return self.__b

    @property
    def A(self):
        """Getter __A"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neuron
        X: numpy.ndarray with shape (nx, m) containing input data
        """
        Z = np.matmul(self.__W, X) + self.__b
        # Sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates model cost using logistic regression
        Y: numpy.ndarray with shape (1, m) containing correct labels
        A: numpy.ndarray with shape (1, m) containing activated outputs
        """
        m = Y.shape[1]
        # Use 1.0000001 - A to avoid division by zero
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates neuron predictions
        X: numpy.ndarray (nx, m) containing input data
        Y: numpy.ndarray (1, m) containing correct labels
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Threshold 0.5 binary classification
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X: numpy.ndarray (nx, m) containing input data
        Y: numpy.ndarray (1, m) containing correct labels
        A: numpy.ndarray (1, m) containing activated output
        alpha: the learning rate
        """
        m = Y.shape[1]
        dz = A - Y
        # X.T aligns shapes: (1, m) * (m, nx) = (1, nx)
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

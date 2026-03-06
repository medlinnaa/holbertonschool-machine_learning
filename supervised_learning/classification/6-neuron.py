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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron using gradient descent
        X: numpy.ndarray (nx, m) containing input data
        Y: numpy.ndarray (1, m) containing correct labels
        iterations: number of iterations to train over
        alpha: the learning rate
        Returns: the evaluation of the training data after training
        """
        # 1. Validation of iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # 2. Validation of alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # 3. The Training Loop
        # We use a loop to repeat the process of guessing and correcting
        i = 0
        while i < iterations:
            # Calculate current prediction (A)
            self.forward_prop(X)
            # Adjust weights and bias based on the error
            self.gradient_descent(X, Y, self.__A, alpha)
            i += 1

        # 4. Final Evaluation
        return self.evaluate(X, Y)

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
        _, a2 = self.forward_prop(X)
        cost = self.cost(Y, a2)
        prediction = np.where(a2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        X: numpy.ndarray with shape (nx, m) containing input data
        Y: numpy.ndarray with shape (1, m) containing correct labels
        A1: output of the hidden layer
        A2: predicted output
        alpha: the learning rate
        """
        m = Y.shape[1]

        # Output Layer Gradients
        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # Hidden Layer Gradients
        # Backpropagated error from dz2 multiplied by sigmoid derivative
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # Update private attributes
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        X: numpy.ndarray with shape (nx, m) containing input data
        Y: numpy.ndarray with shape (1, m) containing correct labels
        iterations: number of iterations to train over
        alpha: the learning rate
        Returns: evaluation of training data after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Training Loop - only one loop used
        i = 0
        while i < iterations:
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            i += 1

        return self.evaluate(X, Y)

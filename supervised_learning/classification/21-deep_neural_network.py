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

        for layer in range(1, self.__L + 1):
            nodes = layers[layer - 1]
            if not isinstance(nodes, int) or (
                    nodes <= 0):
                raise TypeError("layers must be a list of positive integers")

            prev_size = nx if layer == 1 else layers[layer - 2]

            val = np.random.randn(nodes, prev_size)
            self.__weights['W' + str(layer)] = val * np.sqrt(2 / prev_size)
            self.__weights['b' + str(layer)] = np.zeros((nodes, 1))

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

    def forward_prop(self, X):
        """
        Calculates forward propagation of the deep neural network
        X: numpy.ndarray with shape (nx, m) containing input data
        Returns: output of the neural network and the cache
        """
        self.__cache['A0'] = X
        for layer in range(1, self.__L + 1):
            w = self.__weights['W' + str(layer)]
            b = self.__weights['b' + str(layer)]
            a_prev = self.__cache['A' + str(layer - 1)]

            z = np.matmul(w, a_prev) + b
            self.__cache['A' + str(layer)] = 1 / (1 + np.exp(-z))

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates model cost using logistic regression
        Y: numpy.ndarray with shape (1, m) containing correct labels
        A: numpy.ndarray with shape (1, m) containing activated outputs
        Returns: the cost
        """
        m = Y.shape[1]
        log_probs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(log_probs)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network predictions
        X: numpy.ndarray with shape (nx, m) containing input data
        Y: numpy.ndarray with shape (1, m) containing correct labels
        Returns: prediction and cost
        """
        a, _ = self.forward_prop(X)
        cost = self.cost(Y, a)
        prediction = np.where(a >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Y: numpy.ndarray with shape (1, m) containing correct labels
        cache: dictionary containing all intermediary values of the network
        alpha: the learning rate
        """
        m = Y.shape[1]
        # Starting dz for the output layer (dz_L)
        dz = cache['A' + str(self.__L)] - Y

        # Moving backward through the layers
        for layer in range(self.__L, 0, -1):
            a_prev = cache['A' + str(layer - 1)]
            w_curr = self.__weights['W' + str(layer)]
            b_curr = self.__weights['b' + str(layer)]

            # Calculate gradients for current layer
            dw = (1 / m) * np.matmul(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            # Propagate dz back to the previous layer (except for layer 1)
            if layer > 1:
                # Chain rule: derivative of cost * derivative of sigmoid
                dz = np.matmul(w_curr.T, dz) * (a_prev * (1 - a_prev))

            # Update weights and biases for the current layer
            self.__weights['W' + str(layer)] = w_curr - (alpha * dw)
            self.__weights['b' + str(layer)] = b_curr - (alpha * db)

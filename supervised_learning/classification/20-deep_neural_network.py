#!/usr/bin/env python3
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
            # Condition broken into two lines to stay under 80 characters
            if not isinstance(nodes, int) or (
                    nodes <= 0):
                raise TypeError("layers must be a list of positive integers")

            prev_size = nx if layer == 1 else layers[layer - 2]

            # Weights initialization using He et al. method
            val = np.random.randn(nodes, prev_size)
            self.__weights['W' + str(layer)] = val * np.sqrt(2 / prev_size)

            # Bias initialized to zeros
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

        # Single loop to move through all layers
        for layer in range(1, self.__L + 1):
            w = self.__weights['W' + str(layer)]
            b = self.__weights['b' + str(layer)]
            a_prev = self.__cache['A' + str(layer - 1)]

            # Calculate Z and apply Sigmoid activation
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
        # Using 1.0000001 - A to avoid division by zero errors
        log_probs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(log_probs)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network's predictions
        X: numpy.ndarray with shape (nx, m) containing input data
        Y: numpy.ndarray with shape (1, m) containing correct labels
        Returns: prediction and cost
        """
        # Step 1: Forward Propagation to get the last layer's output (A)
        # We don't need the cache for evaluation, so we use _
        A, _ = self.forward_prop(X)

        # Step 2: Calculate the cost based on the final output
        cost = self.cost(Y, A)

        # Step 3: Convert probabilities (A) to binary labels (0 or 1)
        # If A >= 0.5, label is 1; else, label is 0
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

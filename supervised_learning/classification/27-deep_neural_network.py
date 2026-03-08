#!/usr/bin/env python3
"""Defines a deep neural network for multiclass classification"""
import numpy as np
import os


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network performing multiclass classification
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network
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
            if not isinstance(nodes, int) or (nodes <= 0):
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
        Calculates forward propagation with Softmax in the last layer
        """
        self.__cache['A0'] = X
        for layer in range(1, self.__L + 1):
            w = self.__weights['W' + str(layer)]
            b = self.__weights['b' + str(layer)]
            a_prev = self.__cache['A' + str(layer - 1)]
            z = np.matmul(w, a_prev) + b

            if layer < self.__L:
                # Sigmoid for hidden layers
                self.__cache['A' + str(layer)] = 1 / (1 + np.exp(-z))
            else:
                # Softmax for output layer
                exp_z = np.exp(z)
                self.__cache['A' + str(layer)] = exp_z / np.sum(
                    exp_z, axis=0, keepdims=True)

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates multiclass categorical cross-entropy cost
        """
        m = Y.shape[1]
        # Sum log-probabilities only for the correct classes
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network by predicting one-hot results
        """
        a, _ = self.forward_prop(X)
        cost = self.cost(Y, a)
        # Find the index of the highest probability per column
        max_indices = np.argmax(a, axis=0)
        # Convert those indices back to a one-hot matrix
        prediction = np.zeros(a.shape)
        prediction[max_indices, np.arange(a.shape[1])] = 1
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""
        m = Y.shape[1]
        dz = cache['A' + str(self.__L)] - Y
        for layer in range(self.__L, 0, -1):
            a_prev = cache['A' + str(layer - 1)]
            w_curr = self.__weights['W' + str(layer)]
            b_curr = self.__weights['b' + str(layer)]
            dw = (1 / m) * np.matmul(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            if layer > 1:
                dz = np.matmul(w_curr.T, dz) * (a_prev * (1 - a_prev))
            self.__weights['W' + str(layer)] = w_curr - (alpha * dw)
            self.__weights['b' + str(layer)] = b_curr - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the multiclass network"""
        import matplotlib.pyplot as plt
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs, iter_list = [], []
        for i in range(iterations + 1):
            if i < iterations:
                a_last, _ = self.forward_prop(X)
            else:
                a_last = self.__cache['A' + str(self.__L)]
            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, a_last)
                if verbose:
                    print("Cost after {} iterations: {}"
                          .format(i, current_cost))
                if graph:
                    costs.append(current_cost)
                    iter_list.append(i)
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.plot(iter_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves object to pickle file"""
        import pickle
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads object from pickle file"""
        import pickle
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            try:
                return pickle.load(f)
            except Exception:
                return None

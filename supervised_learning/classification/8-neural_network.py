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
        # 1. Validation of nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # 2. Validation of nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # 3. Hidden Layer Attributes
        # W1: (nodes, nx), b1: (nodes, 1)
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # 4. Output Layer Attributes
        # W2: (1, nodes), b2: 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0

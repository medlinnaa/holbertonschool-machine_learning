#!/usr/bin/env python3
"""defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """class that defines a single neuron"""

    def __init__(self, nx):
        """
        initializes the neuron
        nx is the number of input features
        """
        # 1. Validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # 2. Public instance attributes
        # W initialized with random normal distribution in a 2D shape (1, nx)
        self.W = np.random.randn(1, nx)

        # b initialized to 0
        self.b = 0

        # A (activated output) initialized to 0
        self.A = 0

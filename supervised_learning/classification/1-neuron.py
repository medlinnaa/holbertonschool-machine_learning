#!/usr/bin/env python3
"""Defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class that defines a single neuron with private attributes"""

    def __init__(self, nx):
        """Initializes the neuron with private attributes"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Private attributes (double underscore)
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the private weights vector __W"""
        return self.__W

    @property
    def b(self):
        """Getter for the private bias __b"""
        return self.__b

    @property
    def A(self):
        """Getter for the private activated output __A"""
        return self.__A

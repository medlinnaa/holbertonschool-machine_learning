#!/usr/bin/env python3
"""
Module that contains the GaussianProcess class.
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the GaussianProcess instance.

        Args:
            X_init (numpy.ndarray): shape (t, 1) representing the inputs
                already sampled with the black-box function.
            Y_init (numpy.ndarray): shape (t, 1) representing the outputs
                of the black-box function for each input in X_init.
            l (float, int): the length parameter for the kernel.
            sigma_f (float, int): the standard deviation given to the output
                of the black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).

        Args:
            X1 (numpy.ndarray): shape (m, 1).
            X2 (numpy.ndarray): shape (n, 1).

        Returns:
            numpy.ndarray: the covariance kernel matrix of shape (m, n).
        """
        # Using broadcasting to find the squared Euclidean distance
        # between each point in X1 and X2
        sq_dist = (X1 - X2.T) ** 2

        # Applying the RBF formula
        K = (self.sigma_f ** 2) * np.exp(-0.5 * sq_dist / (self.l ** 2))

        return K

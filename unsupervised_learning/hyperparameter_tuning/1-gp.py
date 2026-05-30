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
        sq_dist = (X1 - X2.T) ** 2
        K = (self.sigma_f ** 2) * np.exp(-0.5 * sq_dist / (self.l ** 2))
        return K

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): shape (s, 1) containing all of the points
                whose mean and standard deviation should be calculated.

        Returns:
            tuple: (mu, sigma)
                mu (numpy.ndarray): shape (s,) containing the mean for each
                    point in X_s.
                sigma (numpy.ndarray): shape (s,) containing the variance
                    for each point in X_s.
        """
        # K_s is the covariance between the observed points and the new points
        K_s = self.kernel(self.X, X_s)

        # K_ss is the covariance among the new points
        K_ss = self.kernel(X_s, X_s)

        # Inverse of the covariance matrix of the observed points
        K_inv = np.linalg.inv(self.K)

        # Calculate the predictive mean: mu = K_s.T * K^-1 * Y
        mu = K_s.T.dot(K_inv).dot(self.Y)
        mu = mu.reshape(-1)  # Flatten to shape (s,)

        # Calculate the predictive covariance matrix
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        # Extract the variance (the diagonal of the covariance matrix)
        sigma = np.diag(cov_s)

        return mu, sigma

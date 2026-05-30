#!/usr/bin/env python3
"""
Module that contains the BayesianOptimization class.
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the BayesianOptimization instance.

        Args:
            f (function): the black-box function to be optimized.
            X_init (numpy.ndarray): shape (t, 1) representing the inputs
                already sampled with the black-box function.
            Y_init (numpy.ndarray): shape (t, 1) representing the outputs
                of the black-box function for each input in X_init.
            bounds (tuple): (min, max) representing the bounds of the space
                in which to look for the optimal point.
            ac_samples (int): the number of samples that should be analyzed
                during acquisition.
            l (float, int): the length parameter for the kernel.
            sigma_f (float, int): the standard deviation given to the output
                of the black-box function.
            xsi (float): the exploration-exploitation factor for acquisition.
            minimize (bool): determines whether optimization should be
                performed for minimization (True) or maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using the Expected
        Improvement (EI) acquisition function.

        Returns:
            tuple: (X_next, EI)
                X_next (numpy.ndarray): shape (1,) representing the next
                    best sample point.
                EI (numpy.ndarray): shape (ac_samples,) containing the expected
                    improvement of each potential sample.
        """
        # Get the predictive mean and standard deviation for the sample points
        mu, sigma = self.gp.predict(self.X_s)

        # Determine the current optimal value depending on minimize flag
        if self.minimize is True:
            Y_opt = np.min(self.gp.Y)
            # The improvement is positive when mu is lower than the current min
            imp = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            # The improvement is positive
            # when mu is higher than the current max
            imp = mu - Y_opt - self.xsi

        # Initialize Z and EI arrays with zeros to safely handle sigma == 0
        Z = np.zeros(sigma.shape)
        EI = np.zeros(sigma.shape)

        # Calculate Z and EI only where sigma > 0 to prevent division by zero
        Z[sigma > 0] = imp[sigma > 0] / sigma[sigma > 0]

        term1 = imp[sigma > 0] * norm.cdf(Z[sigma > 0])
        term2 = sigma[sigma > 0] * norm.pdf(Z[sigma > 0])
        EI[sigma > 0] = term1 + term2

        # Find the point in X_s that maximizes the Expected Improvement
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

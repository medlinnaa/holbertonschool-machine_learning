#!/usr/bin/env python3
"""
Module to calculate a GMM from a dataset using sklearn.
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset using sklearn.mixture.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        k (int): The number of clusters.

    Returns:
        tuple: (pi, m, S, clss, bic)
            pi (numpy.ndarray): Cluster priors.
            m (numpy.ndarray): Centroid means.
            S (numpy.ndarray): Covariance matrices.
            clss (numpy.ndarray): Cluster indices of each point.
            bic (float): BIC value of the model.
    """
    # Initialize the Gaussian Mixture model
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    # Fit the model to the dataset
    gmm_model.fit(X)

    # Extract the required parameters directly from the trained model
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_

    # Predict the cluster assignments
    clss = gmm_model.predict(X)

    # Calculate the Bayesian Information Criterion
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic

#!/usr/bin/env python3
"""Module to calculate the definiteness of a square matrix"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a numpy.ndarray
    Returns: string description of definiteness or None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except Exception:
        return None

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    return None

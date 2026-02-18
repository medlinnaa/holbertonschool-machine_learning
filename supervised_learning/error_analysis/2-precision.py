#!/usr/bin/env python3
"""Module to calculate precision
for each class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """
    function just calculates the precision
    for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
            row indices = correct labels
            column indices = predicted labels

    Returns:
        A numpy.ndarray of shape (classes,) containing precision of each class
    """
    true_positives = np.diag(confusion)

    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives

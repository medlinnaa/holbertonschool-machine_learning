#!/usr/bin/env python3
"""Module to calculate sensitivity
aka recall aka
true positive rate for each class"""
import numpy as np


def sensitivity(confusion):
    """
    here the function calculates recall
    for each class in a confusion matrix.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
            row indices = correct labels
            column indices = predicted labels

    Returns:
        A numpy.ndarray of shape (classes,)
        containing sensitiviy of each class
    """
    true_positives = np.diag(confusion)

    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives

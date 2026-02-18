#!/usr/bin/env python3
"""Module to calculate speficity"""
import numpy as npy


def speicificity(confusion):
    """
    this function calculates the specificity
    for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
            row indices = correct labels
            column indices = predicted labels

    Returns:
        A numpy.ndarray of shape (classes,) containing specificity of each class
    """
    tp = np.diag(confusion)

    actual_positives = np.sum(confusion, axis=1)

    predicted_positives = np.sum(confusion, axis=0)

    total_samples = np.sum(confusion)

    tn = total samples - (actual_positives + predicted_positives - tp)

    fp = predicted_positives - tp

    return tn / (tn + fp)

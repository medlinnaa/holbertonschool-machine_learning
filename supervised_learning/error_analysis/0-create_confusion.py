#!/usr/bin/env python3
"""Module to create a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    this function creates confusion matrix
    
    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
        logits: one-hot numpy.ndarray of shape (m, classes)
        
    Returns:
        confusion: numpy.ndarray of shape (classes, classes)
    """

    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)

    classes = labels.shape[1]

    confusion = np.zeros((classes, classes))
 
    for i in range(len(y_true)):
        confusion[y_true[i]][y_pred[i]] += 1

    return confusion

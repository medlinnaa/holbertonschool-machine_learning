#!/usr/bin/env python3
"""Module to update variables using Adam optimization"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm
    Args:
        alpha: the learning rate
        beta1: weight for the first moment
        beta2: weight for the second moment
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var
        s: previous second moment of var
        t: time step used for bias correction
    Returns: updated variable, new first moment, and new second moment
    """
    # 1. Update moments
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # 2. Bias correction
    v_corr = v_new / (1 - (beta1 ** t))
    s_corr = s_new / (1 - (beta2 ** t))

    # 3. Update variable
    var -= alpha * (v_corr / (np.sqrt(s_corr) + epsilon))

    return var, v_new, s_new

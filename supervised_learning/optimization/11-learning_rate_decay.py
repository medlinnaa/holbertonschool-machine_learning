#!/usr/bin/env python3
"""Module to calculate learning rate decay in a stepwise fashion"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy
    Args:
        alpha: the original learning rate
        decay_rate: determines the rate at which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes before alpha is decayed further
    Returns: the updated value for alpha
    """
    # Calculate the staircase step using floor division
    step = global_step // decay_step

    # Apply the inverse time decay formula
    alpha_decayed = alpha / (1 + decay_rate * step)

    return alpha_decayed

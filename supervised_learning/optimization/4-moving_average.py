#!/usr/bin/env python3
"""Module to calculate moving average with bias correction"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
    Args:
        data: list of data to calculate the moving average of
        beta: the weight used for the moving average
    Returns: a list containing the moving averages of data
    """
    v = 0
    m_averages = []

    for i, theta in enumerate(data):
        # Calculate the moving average
        v = beta * v + (1 - beta) * theta

        # Apply bias correction (i + 1 because t starts at 1)
        v_corrected = v / (1 - (beta ** (i + 1)))

        m_averages.append(v_corrected)

    return m_averages

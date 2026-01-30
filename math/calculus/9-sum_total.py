#!/usr/bin/env python3
"""Calculates sum of squares from 1 to n"""


def summation_i_squared(n):
    """Returns the sum of i^2 from i=1 to n, or None if n is invalid"""
    if type(n) is not int or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6

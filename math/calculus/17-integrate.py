#!/usr/bin/env python3
"""Polynomial integration module"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    poly is a list of coefficients representing a polynomial.
    The index of the list represents the power of x.
    C is the integration constant.

    Returns a new list of coefficients representing the integral
    of the polynomial, or None if input is invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None

    for val in poly:
        if not isinstance(val, (int, float)):
            return None

    result = [C]
    power = 1

    for coeff in poly:
        new_coeff = coeff / power

        if new_coeff == int(new_coeff):
            new_coeff = int(new_coeff)

        result.append(new_coeff)
        power += 1

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result

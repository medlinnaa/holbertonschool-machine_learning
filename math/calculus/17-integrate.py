#!/usr/bin/env python3
"""Polynomial integration"""


def poly_integral(poly, C=0):
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

        # convert to int if whole number
        if new_coeff == int(new_coeff):
            new_coeff = int(new_coeff)

        result.append(new_coeff)
        power += 1

    # remove trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result

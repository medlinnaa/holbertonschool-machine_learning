#!/usr/bin/env python3
"""performing element-wise operations on numpy arrays"""


def np_elementwise(mat1, mat2):
    """a function that returns
    sum, difference, product, and quotient element-wise."""
    return (mat1 + mat2,
            mat1 - mat2,
            mat1 * mat2,
            mat1 / mat2)

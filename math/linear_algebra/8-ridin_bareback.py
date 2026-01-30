#!/usr/bin/env python3
"""performing a matrix multiplication."""


def mat_mul(mat1, mat2):
    """a function that returns 
    the product of two 2D matrices, or None if they can't multiply."""
    if len(mat1[0]) != len(mat2):
        return None

    rows_1 = len(mat1)
    cols_1 = len(mat1[0])
    cols_2 = len(mat2[0])

    return [[sum(mat1[i][k] * mat2[k][j] for k in range(cols_1))
             for j in range(cols_2)]
            for i in range(rows_1)]

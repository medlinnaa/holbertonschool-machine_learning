#!/usr/bin/env python3
"""defining a function
to calculate the cofactor matrix."""


def _determinant(matrix):
    """first function is
    a helper function to
    calculate deteminant so that we could find cofactor of the matrix"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * _determinant(sub_matrix)
    return det


def cofactor(matrix):
    """the second function is a function
    that calculates the cofactor matrix of a square matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]

            minor_val = _determinant(sub_matrix)

            row_cofactors.append(((-1) ** (i + j)) * minor_val)

        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix

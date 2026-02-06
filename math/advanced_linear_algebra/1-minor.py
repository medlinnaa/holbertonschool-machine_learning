#!/usr/bin/env python3
"""writing two functions
to return the minor of a given matrix. """


def _determinant(matrix):
    """first function is
    a helper function to calculate determinant."""
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


def minor(matrix):
    """second function is a function that
    calculates minor of the matrix
    using the result of the helper function."""
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

    minor_matrix = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] 
                    for row in (matrix[:i] + matrix[i+1:])]
            row_minors.append(_determinant(sub_matrix))
        minor_matrix.append(row_minors)

    return minor_matrix

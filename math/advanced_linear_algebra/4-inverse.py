#!/usr/bin/env python3
"""defining two function
to determine the inverse of a matrix."""


def _determinant(matrix):
    """using a helper function to calculate matrix determinant."""
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


def inverse(matrix):
    """second function just calculates
    the inverse of a matrix with the help of the first function."""
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

    det = _determinant(matrix)
    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    adj = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]
            adj[j][i] = (((-1) ** (i + j)) * _determinant(sub_matrix)) / det

    return adj

#!/usr/bin/env python3
"""using two functions to find the adjugate of a given matrix."""


def _determinant(matrix):
    """again, a first function that is used as a helper function to calculate matrix determinant"""
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


def adjugate(matrix):
    """second function just gives the answer of adjugate of the given matrix."""
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

    adj = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]
            # Transpose happens here: cofactor(i,j) goes to adj(j,i)
            adj[j][i] = ((-1) ** (i + j)) * _determinant(sub_matrix)

    return adj

#!/usr/bin/env python3
"""defining a function called determinant in order to calculate the determinant of the given matrix. """

def determinant(matrix):
    """this function just returns the determinant of an arbitrary matrix
    by using some conditional statements for verification
    and later it also contains
    the logic of determinant definition to solve it. """
    if matrix == [[]]:
        return 1
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")


    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    else:
        det = 0
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]

            sign = (-1) ** j

            det += sign * matrix[0][j] * determinant(sub_matrix)
    return det

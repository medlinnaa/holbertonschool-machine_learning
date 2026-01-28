#!/usr/bin/env python3
"""defining the function itself """


def matrix_transpose(matrix):
    """ a function that returns 
    the transpose of a matrix using list comprehension """
    row = len(matrix)
    col = len(matrix[0])
    return [[matrix[r][c] for r in range(row)] for c in range(col)]

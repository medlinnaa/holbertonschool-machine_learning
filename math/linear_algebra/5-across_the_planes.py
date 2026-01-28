#!/usr/bin/env python3
"""defining our function below """


def add_matrices2D(mat1, mat2):
    """a function that returns
    a new matrix with element wise sums or
    None if shapes are different"""
    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]

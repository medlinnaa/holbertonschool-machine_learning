#!/usr/bin/env python3
"""Defines a function that returns the shape of a matrix."""


def matrix_shape(matrix):
    """here is the function that returns
   the shape of the matrix as a list of integers"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

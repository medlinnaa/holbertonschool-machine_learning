#!/usr/bin/env python3
"""adding two elements of the arrays """


def add_arrays(arr1, arr2):
    """a function that returns a new list with element-wise sums, or None if shapes differ."""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]

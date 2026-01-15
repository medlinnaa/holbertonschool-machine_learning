#!/usr/bin/env python3
"""2-from_file.py"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pandas DataFrame.

    Args:
        filename (str): file path to load
        delimiter (str): column separator

    Returns:
        pd.DataFrame: loaded dataframe
    """
    return pd.read_csv(filename, delimiter=delimiter)

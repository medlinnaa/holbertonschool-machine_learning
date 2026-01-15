#!/usr/bin/env python3
"""13-analyze.py"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): dataframe containing a Timestamp column

    Returns:
        pd.DataFrame: descriptive statistics dataframe
    """
    return df.drop(columns=["Timestamp"]).describe()

#!/usr/bin/env python3
"""8-prune.py"""


def prune(df):
    """
    Removes entries where Close has NaN values.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    return df.dropna(subset=["Close"])

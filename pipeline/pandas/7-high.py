#!/usr/bin/env python3
"""7-high.py"""


def high(df):
    """
    Sorts the DataFrame by the High column in descending order.

    Returns:
        pd.DataFrame: sorted dataframe
    """
    return df.sort_values("High", ascending=False)

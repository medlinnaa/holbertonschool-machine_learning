#!/usr/bin/env python3
"""5-slice.py"""


def slice(df):
    """
    Extract columns High, Low, Close, and Volume_(BTC),
    then select every 60th row.

    Returns:
        pd.DataFrame: sliced dataframe
    """
    cols = ["High", "Low", "Close", "Volume_(BTC)"]
    return df[cols].iloc[::60]

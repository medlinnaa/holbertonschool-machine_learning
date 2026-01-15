#!/usr/bin/env python3
"""11-concat.py
Concatenates Bitstamp and Coinbase BTC datasets into a single DataFrame
with a MultiIndex identifying the data source.
"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """Concatenate two DataFrames after indexing on Timestamp.

    The function performs the following operations:
    - Indexes both df1 and df2 on their 'Timestamp' columns.
    - Selects all rows from df2 with timestamps up to and including 1417411920.
    - Concatenates the selected rows from df2 above df1.
    - Adds keys to label df2 rows as 'bitstamp' and df1 rows as 'coinbase'.

    Args:
        df1 (pd.DataFrame): Coinbase dataframe containing a 'Timestamp' column.
        df2 (pd.DataFrame): Bitstamp dataframe containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: Concatenated dataframe with MultiIndex keys
        ('bitstamp' and 'coinbase').
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

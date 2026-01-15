#!/usr/bin/env python3
"""12-hierarchy.py
Builds a hierarchical (MultiIndex) DataFrame for a specific time window,
with Timestamp as the first index level and the exchange as the second.
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """Create a hierarchical DataFrame with Timestamp as the first index level.

    The function:
    - Indexes both dataframes on their 'Timestamp' column.
    - Selects rows from timestamps 1417411980 to 1417417980 (inclusive).
    - Concatenates df2 (bitstamp) and df1 (coinbase) with keys.
    - Rearranges the MultiIndex so Timestamp is level 0 and exchange is level 1.
    - Sorts to ensure chronological order.

    Args:
        df1 (pd.DataFrame): Coinbase dataframe with a 'Timestamp' column.
        df2 (pd.DataFrame): Bitstamp dataframe with a 'Timestamp' column.

    Returns:
        pd.DataFrame: Concatenated DataFrame indexed by (Timestamp, exchange).
    """
    df1 = index(df1).loc[1417411980:1417417980]
    df2 = index(df2).loc[1417411980:1417417980]

    df = pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )

    # Current index: (exchange, Timestamp) -> swap to (Timestamp, exchange)
    df = df.swaplevel(0, 1).sort_index()

    return df

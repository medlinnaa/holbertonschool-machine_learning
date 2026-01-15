#!/usr/bin/env python3
"""6-flip_switch.py"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order (newest first),
    then transposes it.

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df = df.sort_values("Timestamp", ascending=False)
    return df.T

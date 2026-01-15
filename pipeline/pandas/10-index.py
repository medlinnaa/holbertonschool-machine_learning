#!/usr/bin/env python3
"""10-index.py"""


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame.

    Returns:
        pd.DataFrame: modified dataframe
    """
    return df.set_index("Timestamp")

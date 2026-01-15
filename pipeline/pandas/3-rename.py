#!/usr/bin/env python3
"""3-rename.py"""

import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime, converts it to datetime,
    and returns only Datetime and Close columns.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]

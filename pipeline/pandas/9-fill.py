#!/usr/bin/env python3
"""9-fill.py"""


def fill(df):
    """
    Cleans the DataFrame:
    - Removes Weighted_Price column
    - Fills Close NaN with previous row's value
    - Fills Open/High/Low NaN with Close value in same row
    - Fills Volume_(BTC) and Volume_(Currency) NaN with 0
    """
    df = df.drop(columns=["Weighted_Price"])

    # Fill Close with previous row value
    df["Close"] = df["Close"].ffill()

    # Fill Open/High/Low with same-row Close
    df["Open"] = df["Open"].fillna(df["Close"])
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])

    # Fill volumes with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df

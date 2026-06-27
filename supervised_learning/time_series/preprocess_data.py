#!/usr/bin/env python3
"""
Preprocesses Bitcoin data from Coinbase and Bitstamp for time series forecasting.
"""
import pandas as pd


def preprocess_data(file_path, output_path="preprocessed_btc.csv"):
    """
    Loads, cleans, resamples, and saves the BTC dataset.

    Args:
        file_path (str): Path to the raw CSV file.
        output_path (str): Path to save the preprocessed data.
    """
    # Load raw data
    df = pd.read_csv(file_path)

    # Drop Weighted_Price as Close is our primary regression target
    df = df.drop(columns=['Weighted_Price'])

    # Handle missing values by forward-filling prices and zero-filling volume
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # Convert Unix timestamp to datetime and set as the DataFrame index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')

    # Resample the 60-second raw windows into 1-hour blocks
    df_resampled = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    # Drop any edge-case NaNs remaining at the very start of the timeline
    df_resampled = df_resampled.dropna()

    # Save the cleaned and aggregated dataset
    df_resampled.to_csv(output_path)
    print(f"Data preprocessed and saved to {output_path}")


if __name__ == "__main__":
    # Path to your Bitstamp dataset in Google Drive
    bitstamp_path = (
        '/content/drive/MyDrive/'
        'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    )

    # Path to your Coinbase dataset in Google Drive
    coinbase_path = (
        '/content/drive/MyDrive/'
        'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    )

    print("Processing Bitstamp dataset...")
    preprocess_data(bitstamp_path, output_path="preprocessed_bitstamp.csv")

    print("\nProcessing Coinbase dataset...")
    preprocess_data(coinbase_path, output_path="preprocessed_coinbase.csv")

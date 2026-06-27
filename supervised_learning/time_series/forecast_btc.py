#!/usr/bin/env python3
"""
Creates, trains, and validates a Keras model for forecasting BTC.
"""
import pandas as pd
import tensorflow as tf


def create_dataset(data, lookback=24):
    """
    Creates a tf.data.Dataset for time series forecasting.

    Args:
        data (numpy.ndarray): The scaled dataset.
        lookback (int): The number of past hours to use as input.

    Returns:
        tf.data.Dataset: The formatted dataset.
    """
    inputs = data[:-1]
    targets = data[lookback:, 3]  # Close price is at index 3

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=inputs,
        targets=targets,
        sequence_length=lookback,
        batch_size=64,
        shuffle=True
    )
    return dataset


def build_model(input_shape):
    """
    Builds and compiles an LSTM model.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, features).

    Returns:
        tf.keras.Model: The compiled model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    # 1. Load preprocessed data
    df = pd.read_csv("preprocessed_coinbase.csv", index_col="Timestamp")

    # 2. Scale data using simple min-max normalization
    data_min = df.min()
    data_max = df.max()
    scaled_data = (df - data_min) / (data_max - data_min)
    scaled_data = scaled_data.values

    # 3. Train/Validation Split (80/20)
    split_idx = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_idx]
    val_data = scaled_data[split_idx:]

    # 4. Create tf.data.Dataset instances
    lookback_window = 24
    train_ds = create_dataset(train_data, lookback_window)
    val_ds = create_dataset(val_data, lookback_window)

    # 5. Build, train, and save model
    in_shape = (lookback_window, scaled_data.shape[1])
    model = build_model(in_shape)

    print("Training model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    model.save('btc_model.h5')
    print("Model saved to btc_model.h5")

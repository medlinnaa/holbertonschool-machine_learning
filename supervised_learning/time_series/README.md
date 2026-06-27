# Time Series Forecasting

> Implementation of a Bitcoin (BTC) price forecasting model using TensorFlow/Keras and Recurrent Neural Networks (RNNs).


## 📖Description

This project builds a complete deep learning pipeline to forecast short-term Bitcoin prices using raw financial datasets from the Coinbase and Bitstamp exchanges. 

Because financial markets produce massive amounts of raw, irregular data, the system first cleans and structures minute-by-minute market activity into stable, hourly intervals. It then feeds this sequential data into a neural network designed to understand the context of time. By analyzing the market trends, volume, and price fluctuations of the past 24 hours, the model outputs a predicted close price for the subsequent hour, simulating the average timeframe of a BTC transaction.


## 🧠Concepts

* Data Resampling & Imputation: Transforming irregular 60-second time windows into clean 1-hour intervals by forward-filling missing prices and aggregating trading volumes.
* Sequence Generation: Utilizing `tf.data.Dataset` to slice continuous time-series data into discrete sliding windows (24 steps of input mapped to 1 step of target output).
* Long Short-Term Memory (LSTM): Implementing a specialized RNN architecture that uses internal gating mechanisms to remember important historical market signals while forgetting irrelevant noise.
* Feature Normalization: Applying min-max scaling to standardize massive financial metrics (like USD volume and BTC prices) to ensure stable and efficient gradient descent during training.


## ⚙️Requirements

* Python 3.9
* TensorFlow 2.15
* NumPy 1.25.2
* Pandas 2.2.2
* Ubuntu 20.04 LTS

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp -> Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamps to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index on Date
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].ffill()

for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col] = df[col].fillna(0)

# Keep 2017 and beyond, then resample daily with required aggregations
df = df.loc['2017-01-01':].resample('D').agg(
    {
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
    }
)

# Return the transformed df before plotting (in script: keep df, print it)
print(df)

# Plot
df.plot()
plt.show()

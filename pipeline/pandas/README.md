# Pandas

> Implementation of data cleaning, structural manipulation, and time-series analysis using Python's Pandas library.


## 📖Description

This project builds a structured data processing pipeline for financial datasets, specifically focusing on historical Bitcoin exchange records from Coinbase and Bitstamp. It walks through the practical stages of preparing raw, unrefined data for high-level analysis. 

The scripts form a toolkit designed to ingest information from various origins, normalize inconsistent formatting, and seamlessly merge multiple datasets. Rather than simply deleting incomplete data, the pipeline focuses on intelligent data recovery and preparation, ultimately culminating in the aggregation of cleaned time-series data to visualize long-term market trends.


## 🧠Concepts

* Data Ingestion: Instantiating DataFrames from multi-dimensional NumPy arrays, native Python dictionaries, and external CSV files.
* Time-Series Operations: Translating Unix epoch timestamps into standard datetime objects, establishing chronological indices, and resampling data into specific temporal frequencies (such as daily intervals) using targeted aggregation metrics (max, min, mean, sum).
* Structural Transformations: Executing matrix transpositions, multi-level sorting algorithms, incremental row slicing, and constructing hierarchical MultiIndexes to isolate and query data by source exchange.
* Data Imputation & Pruning: Resolving `NaN` values safely by applying forward-filling techniques and localized row-value mapping, combined with strategic column dropping to maintain data integrity.
* Statistical Summarization: Extracting descriptive statistics automatically to evaluate the variance, bounds, and central tendencies of large financial datasets without manual computation.


## ⚙️Requirements

* Python 3.9
* Pandas
* NumPy
* Matplotlib
* Ubuntu 20.04

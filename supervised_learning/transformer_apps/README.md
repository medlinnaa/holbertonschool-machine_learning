# Transformer Applications

> Implementation of a machine translation model from Portuguese to English using a custom Transformer architecture.


## 📖Description

This project constructs a comprehensive data pipeline and deep learning model designed to translate text from Portuguese to English.

Operating on a parallel corpus extracted from TED talks, the system manages the entire lifecycle of the data, from raw strings to a fully optimized training loop. Instead of relying on traditional recurrent neural networks, the project leverages the attention mechanism to understand the contextual relationship between words in a sentence. The end result is a standalone model that sequentially predicts translated text based on the learned contextual weights of the input sequence.


## 🧠Concepts

* **Data Pipeline Optimization:** Utilizing the TensorFlow `tf.data` API to efficiently filter, cache, pad, batch, and prefetch massive text datasets to prevent memory bottlenecks during training.
* **Sub-word Tokenization:** Training custom tokenizers to convert standard string vocabulary into manageable integer sequences, establishing strict start and end boundaries for translation generation.
* **Masking Mechanisms:** Creating multi-dimensional look-ahead and padding matrices to ensure the network ignores zero-padding and cannot peer into future tokens during sequence prediction.
* **Attention Architecture:** Building custom Encoder and Decoder layers equipped with Multi-Head Attention and Positional Encoding to capture complex grammatical dependencies without convolutions or recurrence.
* **Custom Training Schedules:** Implementing a specialized learning rate schedule for the Adam optimizer alongside a custom sparse categorical crossentropy loss function that explicitly excludes padded tokens from its calculations.


## ⚙️Requirements

* Python 3.9
* TensorFlow 2.15
* TensorFlow Datasets 4.9.2
* Transformers 4.44.2
* NumPy 1.25.2
* Ubuntu 20.04 LTS

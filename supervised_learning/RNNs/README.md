# Recurrent Neural Networks

> Implementation of Recurrent Neural Networks (Simple RNN, GRU, LSTM) from scratch using NumPy.


## 📖Description

This project implements complete forward propagation pipelines for various Recurrent Neural Network (RNN) architectures from scratch using pure Python and NumPy. 

By avoiding high-level frameworks, this project breaks down the mathematical foundations of sequence modeling. It includes the construction of standard Simple RNN cells, Gated Recurrent Units (GRUs), and Long Short-Term Memory (LSTM) cells. It explores how hidden states and cell states are updated step-by-step through specific gate mechanisms (update, reset, forget, and output gates) to process sequential data, and culminates in a Deep RNN implementation handling multiple stacked layers over time.


## 🧠Concepts

* Recurrent Neural Networks: Designing cells that maintain hidden states to process sequences across multiple time steps.
* Gating Mechanisms: Implementing the internal math of GRUs and LSTMs to understand how networks learn to retain long-term dependencies and avoid the vanishing gradient problem.
* Forward Propagation: Calculating linear combinations, applying non-linear activation functions (tanh, sigmoid, softmax), and routing tensor outputs correctly.
* Deep Sequence Models: Stacking multiple RNN layers, where the hidden state sequence of one layer serves as the input sequence to the next.
* Matrix Operations: Efficiently executing batch sequence processing using NumPy concatenations and dot products.


## ⚙️Requirements

* Python 3.9
* NumPy 1.25.2
* Ubuntu 20.04 LTS
* pycodestyle 2.11.1

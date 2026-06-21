#!/usr/bin/env python3
"""
Module that contains the `rnn` function to perform forward propagation
for a simple Recurrent Neural Network over multiple time steps.
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell (RNNCell): An instance of RNNCell that will be used for
            the forward propagation.
        X (numpy.ndarray): The data to be used, of shape (t, m, i).
            t is the maximum number of time steps.
            m is the batch size.
            i is the dimensionality of the data.
        h_0 (numpy.ndarray): The initial hidden state, of shape (m, h).
            h is the dimensionality of the hidden state.

    Returns:
        tuple: (H, Y)
            H is a numpy.ndarray containing all of the hidden states,
                including the initial hidden state h_0. Shape: (t + 1, m, h)
            Y is a numpy.ndarray containing all of the outputs. Shape: (t, m, o)
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Initialize the array to hold all hidden states
    # t + 1 because we include the initial hidden state h_0 at index 0
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Initialize a list to hold all the outputs
    Y = []

    # Loop through each time step
    for step in range(t):
        # Perform forward prop for the current time step
        # using the hidden state from the previous step and the current data
        h_next, y_step = rnn_cell.forward(H[step], X[step])

        # Save the new hidden state and output
        H[step + 1] = h_next
        Y.append(y_step)

    # Convert the outputs list to a numpy array for the final return
    return H, np.array(Y)

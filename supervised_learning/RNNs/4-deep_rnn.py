#!/usr/bin/env python3
"""
Module that contains the `deep_rnn` function to perform forward propagation
for a deep Recurrent Neural Network.
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells (list): A list of RNNCell instances of length l that will
            be used for the forward propagation.
            l is the number of layers.
        X (numpy.ndarray): The data to be used, of shape (t, m, i).
            t is the maximum number of time steps.
            m is the batch size.
            i is the dimensionality of the data.
        h_0 (numpy.ndarray): The initial hidden state, of shape (l, m, h).
            h is the dimensionality of the hidden state.

    Returns:
        tuple: (H, Y)
            H is a numpy.ndarray containing all of the hidden states.
                Shape: (t + 1, l, m, h)
            Y is a numpy.ndarray containing all of the outputs from the
                last layer. Shape: (t, m, o)
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Initialize the array to hold all hidden states
    # t + 1 to include the initial hidden states at time step 0
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    # Initialize a list to hold the outputs from the final layer
    Y = []

    # Loop through each time step
    for step in range(t):
        # The input for the first layer is the actual sequence data
        x_in = X[step]

        # Loop through each layer in the network
        for lay in range(l):
            # Fetch the previous time step's hidden state for this exact layer
            h_prev = H[step, lay]

            # Perform forward prop for this layer
            h_next, y_step = rnn_cells[lay].forward(h_prev, x_in)

            # Save the calculated hidden state into our H tensor
            H[step + 1, lay] = h_next

            # The input for the *next* layer is the hidden state of *this* layer
            x_in = h_next

            # If we are at the very last layer, record the output 'y'
            if lay == l - 1:
                Y.append(y_step)

    # Convert the list of outputs into a numpy array for the final return
    return H, np.array(Y)

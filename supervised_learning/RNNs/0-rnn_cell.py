#!/usr/bin/env python3
"""
Module that contains the RNNCell class for a simple Recurrent Neural Network.
"""
import numpy as np


class RNNCell:
    """
    Represents a single cell of a simple RNN.
    """

    def __init__(self, i, h, o):
        """
        Initializes the RNNCell instance.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Wh is for concatenated hidden state and input data: shape (h + i, h)
        self.Wh = np.random.normal(size=(h + i, h))

        # Wy is for the output: shape (h, o)
        self.Wy = np.random.normal(size=(h, o))

        # Biases initialized to zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): The previous hidden state of shape (m, h).
            x_t (numpy.ndarray): The data input for the cell of shape (m, i).

        Returns:
            tuple: (h_next, y)
                h_next is the next hidden state.
                y is the output of the cell.
        """
        # Concatenate previous hidden state and input data along axis 1
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Calculate the next hidden state using the tanh activation function
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        # Calculate the linear output
        y_linear = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation function to the output
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y

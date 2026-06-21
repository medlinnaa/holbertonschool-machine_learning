#!/usr/bin/env python3
"""
Module that contains the GRUCell class for a Gated Recurrent Unit.
"""
import numpy as np


class GRUCell:
    """
    Represents a single cell of a Gated Recurrent Unit (GRU).
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRUCell instance.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Update gate weights and biases
        self.Wz = np.random.normal(size=(h + i, h))
        self.bz = np.zeros((1, h))

        # Reset gate weights and biases
        self.Wr = np.random.normal(size=(h + i, h))
        self.br = np.zeros((1, h))

        # Intermediate hidden state weights and biases
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): The previous hidden state, shape (m, h).
            x_t (numpy.ndarray): The data input for the cell, shape (m, i).

        Returns:
            tuple: (h_next, y)
                h_next is the next hidden state.
                y is the output of the cell.
        """
        # Concatenate the previous hidden state and the current input data
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # 1. Update Gate (z_t) with sigmoid activation
        z_linear = np.matmul(concat_input, self.Wz) + self.bz
        z_t = 1 / (1 + np.exp(-z_linear))

        # 2. Reset Gate (r_t) with sigmoid activation
        r_linear = np.matmul(concat_input, self.Wr) + self.br
        r_t = 1 / (1 + np.exp(-r_linear))

        # 3. Intermediate Hidden State / Candidate (h_tilde)
        # Apply the reset gate to the previous hidden state before concat
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_linear = np.matmul(concat_reset, self.Wh) + self.bh
        h_tilde = np.tanh(h_linear)

        # 4. Next Hidden State (h_next)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # 5. Output (y) using Softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y

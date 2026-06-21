#!/usr/bin/env python3
"""
Module that contains the LSTMCell class for a Long Short-Term Memory unit.
"""
import numpy as np


class LSTMCell:
    """
    Represents a single cell of an LSTM network.
    """

    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell instance.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Forget gate weights and biases
        self.Wf = np.random.normal(size=(h + i, h))

        # Update gate weights and biases
        self.Wu = np.random.normal(size=(h + i, h))

        # Intermediate cell state weights and biases
        self.Wc = np.random.normal(size=(h + i, h))

        # Output gate weights and biases
        self.Wo = np.random.normal(size=(h + i, h))

        # Outputs weights and biases
        self.Wy = np.random.normal(size=(h, o))

        # Initializing biases to zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): The previous hidden state, shape (m, h).
            c_prev (numpy.ndarray): The previous cell state, shape (m, h).
            x_t (numpy.ndarray): The data input for the cell, shape (m, i).

        Returns:
            tuple: (h_next, c_next, y)
                h_next is the next hidden state.
                c_next is the next cell state.
                y is the output of the cell.
        """
        # Concatenate the previous hidden state and the current input data
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # 1. Forget Gate (f_t)
        f_linear = np.matmul(concat_input, self.Wf) + self.bf
        f_t = 1 / (1 + np.exp(-f_linear))

        # 2. Update Gate (u_t)
        u_linear = np.matmul(concat_input, self.Wu) + self.bu
        u_t = 1 / (1 + np.exp(-u_linear))

        # 3. Intermediate Cell State (c_tilde)
        c_linear = np.matmul(concat_input, self.Wc) + self.bc
        c_tilde = np.tanh(c_linear)

        # 4. Next Cell State (c_next)
        c_next = f_t * c_prev + u_t * c_tilde

        # 5. Output Gate (o_t)
        o_linear = np.matmul(concat_input, self.Wo) + self.bo
        o_t = 1 / (1 + np.exp(-o_linear))

        # 6. Next Hidden State (h_next)
        h_next = o_t * np.tanh(c_next)

        # 7. Output (y) using Softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, c_next, y

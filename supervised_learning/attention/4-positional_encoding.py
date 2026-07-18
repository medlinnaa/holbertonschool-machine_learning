#!/usr/bin/env python3
"""
Positional Encoding module
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer model.

    Args:
        max_seq_len: An integer representing the maximum sequence length.
        dm: An integer representing the model depth (dimensionality).

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    pe = np.zeros((max_seq_len, dm))
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(0, dm, 2)

    # Calculate the denominator: 10000^(2i / dm)
    denominator = np.power(10000, i / dm)

    # Apply sin to even indices in the array; 2i
    pe[:, 0::2] = np.sin(pos / denominator)

    # Apply cos to odd indices in the array; 2i+1
    pe[:, 1::2] = np.cos(pos / denominator)

    return pe

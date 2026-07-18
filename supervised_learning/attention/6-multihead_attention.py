#!/usr/bin/env python3
"""
MultiHead Attention Module
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class that performs multi-head attention.
    """

    def __init__(self, dm, h):
        """
        Class constructor for MultiHeadAttention.

        Args:
            dm: an integer representing the dimensionality of the model
            h: an integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Calculates the multi-head attention.

        Args:
            Q: a tensor of shape (batch, seq_len_q, dk) containing the input
               to generate the query matrix
            K: a tensor of shape (batch, seq_len_v, dk) containing the input
               to generate the key matrix
            V: a tensor of shape (batch, seq_len_v, dv) containing the input
               to generate the value matrix
            mask: always None (or a tensor to apply optional masking)

        Returns:
            output: a tensor with its last two dimensions as
            (..., seq_len_q, dm) containing the scaled dot product attention
            weights: a tensor with its last three dimensions as
                     (..., h, seq_len_q, seq_len_v)
            containing attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Generate query, key, and value matrices
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Split the last dimension into (h, depth)
        q = tf.reshape(q, (batch_size, -1, self.h, self.depth))
        k = tf.reshape(k, (batch_size, -1, self.h, self.depth))
        v = tf.reshape(v, (batch_size, -1, self.h, self.depth))

        # Transpose to shape (batch_size, h, seq_len, depth)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Calculate scaled dot product attention
        output, weights = sdp_attention(q, k, v, mask)

        # Transpose the output back to (batch_size, seq_len_q, h, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])

        # Concatenate the heads back together
        concat_output = tf.reshape(output, (batch_size, -1, self.dm))

        # Pass the concatenated output through the final linear layer
        final_output = self.linear(concat_output)

        return final_output, weights

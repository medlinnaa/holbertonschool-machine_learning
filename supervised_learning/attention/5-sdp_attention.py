#!/usr/bin/env python3
"""
Scaled Dot Product Attention module
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q: a tensor with its last two dimensions as (..., seq_len_q, dk)
           containing the query matrix
        K: a tensor with its last two dimensions as (..., seq_len_v, dk)
           containing the key matrix
        V: a tensor with its last two dimensions as (..., seq_len_v, dv)
           containing the value matrix
        mask: a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
              containing the optional mask, or defaulted to None

    Returns:
        output: a tensor with its last two dimensions as (..., seq_len_q, dv)
                containing the scaled dot product attention
        weights: a tensor with its last two dimensions as
                 (..., seq_len_q, seq_len_v) containing the attention weights
    """
    # Calculate the dot product of Q and K^T
    # transpose_b=True transposes the last two dimensions of K
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale the dot product by the square root of the depth (dk)
    dk = tf.cast(tf.shape(Q)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the mask if it is provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply a softmax function to get the attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the attention weights by V to get the output
    output = tf.matmul(weights, V)

    return output, weights

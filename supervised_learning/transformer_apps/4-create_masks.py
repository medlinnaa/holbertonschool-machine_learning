#!/usr/bin/env python3
"""
Module to create masks for transformer training and validation.
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs (tf.Tensor): A tensor of shape (batch_size, seq_len_in)
            that contains the input sentence.
        target (tf.Tensor): A tensor of shape (batch_size, seq_len_out)
            that contains the target sentence.

    Returns:
        tuple: (encoder_mask, combined_mask, decoder_mask)
            - encoder_mask: padding mask for the encoder of shape
              (batch_size, 1, 1, seq_len_in).
            - combined_mask: max between lookahead and target padding mask
              of shape (batch_size, 1, seq_len_out, seq_len_out).
            - decoder_mask: padding mask for the decoder's 2nd attention
              block of shape (batch_size, 1, 1, seq_len_in).
    """
    # 1. Encoder Padding Mask
    # Marks padded tokens (0) with a 1, otherwise 0
    # Shape: (batch_size, 1, 1, seq_len_in)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # 2. Decoder Padding Mask
    # Identical to the encoder mask, used in the 2nd attention block
    # Shape: (batch_size, 1, 1, seq_len_in)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # 3. Combined Mask (Look-ahead + Target padding)
    # Target padding mask (Shape: (batch_size, 1, 1, seq_len_out))
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Look-ahead mask (Shape: (seq_len_out, seq_len_out))
    # Uses band_part to create a lower triangular matrix, subtracted from 1
    # to create an upper triangular matrix (hiding future tokens)
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )

    # Combine by taking the maximum between the two masks
    # Shape broadcasts to: (batch_size, 1, seq_len_out, seq_len_out)
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask

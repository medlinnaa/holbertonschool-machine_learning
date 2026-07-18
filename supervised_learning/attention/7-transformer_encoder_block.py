#!/usr/bin/env python3
"""
Transformer Encoder Block module
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    EncoderBlock class that creates an encoder block for a transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor for the EncoderBlock.

        Args:
            dm: an integer representing the dimensionality of the model
            h: an integer representing the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate (default is 0.1)
        """
        super(EncoderBlock, self).__init__()

        # Multi-Head Attention layer
        self.mha = MultiHeadAttention(dm, h)

        # Feed-forward network dense layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer normalization layers with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Calculates the output of the encoder block.

        Args:
            x: a tensor of shape (batch, input_seq_len, dm) containing the
               input to the encoder block
            training: a boolean to determine if the model is training
            mask: the mask to be applied for multi-head attention

        Returns:
            A tensor of shape (batch, input_seq_len, dm) containing the
            block's output.
        """
        # Multi-head self-attention: query, key, and value are all x
        attn_output, _ = self.mha(x, x, x, mask)

        # Apply dropout to the attention output
        attn_output = self.dropout1(attn_output, training=training)

        # Residual connection 1 and layer normalization
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)

        # Apply dropout to the feed-forward output
        ffn_output = self.dropout2(ffn_output, training=training)

        # Residual connection 2 and layer normalization
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

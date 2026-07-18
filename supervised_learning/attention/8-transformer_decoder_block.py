#!/usr/bin/env python3
"""
Transformer Decoder Block module
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    DecoderBlock class that creates a decoder block for a transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor for the DecoderBlock.

        Args:
            dm: an integer representing the dimensionality of the model
            h: an integer representing the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate (default is 0.1)
        """
        super(DecoderBlock, self).__init__()

        # Multi-Head Attention layers
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        # Feed-forward network dense layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer normalization layers with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Calculates the output of the decoder block.

        Args:
            x: a tensor of shape (batch, target_seq_len, dm) containing the
               input to the decoder block
            encoder_output: a tensor of shape (batch, input_seq_len, dm)
                            containing the output of the encoder
            training: a boolean to determine if the model is training
            look_ahead_mask: the mask to be applied to the first multi head
                             attention layer
            padding_mask: the mask to be applied to the second multi head
                          attention layer

        Returns:
            A tensor of shape (batch, target_seq_len, dm) containing the
            block's output.
        """
        # Block 1: Masked multi-head self-attention
        # Query, key, and value are all x
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Block 2: Multi-head cross-attention
        # Query is out1, key and value are encoder_output
        attn2, _ = self.mha2(out1, encoder_output, encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Block 3: Feed-forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

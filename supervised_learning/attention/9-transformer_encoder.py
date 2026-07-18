#!/usr/bin/env python3
"""
Transformer Encoder module
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class that creates the encoder for a transformer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor for the Encoder.

        Args:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            input_vocab: the size of the input vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm

        # Embedding layer for the inputs
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)

        # Positional encoding numpy.ndarray
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List of length N containing all of the EncoderBlocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Calculates the output of the encoder.

        Args:
            x: a tensor of shape (batch, input_seq_len) containing the input
               to the encoder
            training: a boolean to determine if the model is training
            mask: the mask to be applied for multi-head attention

        Returns:
            A tensor of shape (batch, input_seq_len, dm) containing the
            encoder output.
        """
        # Get the sequence length of the current input
        seq_len = tf.shape(x)[1]

        # Pass the input through the embedding layer
        x = self.embedding(x)

        # Scale the embeddings by the square root of the model dimensionality
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encodings (cast to float32 to avoid type errors)
        pos_encoding = tf.cast(self.positional_encoding[:seq_len, :],
                               dtype=tf.float32)
        x += pos_encoding

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass the output through all N encoder blocks
        for block in self.blocks:
            x = block(x, training, mask)

        return x

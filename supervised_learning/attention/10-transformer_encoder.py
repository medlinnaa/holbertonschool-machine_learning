#!/usr/bin/env python3
"""
Transformer Decoder module
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Decoder class that creates the decoder for a transformer.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor for the Decoder.

        Args:
            N: the number of blocks in the decoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            target_vocab: the size of the target vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm

        # Embedding layer for the targets
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)

        # Positional encoding numpy.ndarray
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List of length N containing all of the DecoderBlocks
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Calculates the output of the decoder.

        Args:
            x: a tensor of shape (batch, target_seq_len) containing the input
               to the decoder
            encoder_output: a tensor of shape (batch, input_seq_len, dm)
                            containing the output of the encoder
            training: a boolean to determine if the model is training
            look_ahead_mask: the mask to be applied to the first multi head
                             attention layer
            padding_mask: the mask to be applied to the second multi head
                          attention layer

        Returns:
            A tensor of shape (batch, target_seq_len, dm) containing the
            decoder output.
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

        # Pass the output through all N decoder blocks
        for block in self.blocks:
            x = block(x, encoder_output, training,
                      look_ahead_mask, padding_mask)

        return x

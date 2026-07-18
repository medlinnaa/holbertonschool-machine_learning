#!/usr/bin/env python3
"""
Transformer module
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer class that creates a full transformer network.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor for the Transformer.

        Args:
            N: the number of blocks in the encoder and decoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layers
            input_vocab: the size of the input vocabulary
            target_vocab: the size of the target vocabulary
            max_seq_input: the maximum sequence length possible for the input
            max_seq_target: the maximum sequence length possible for the target
            drop_rate: the dropout rate (default is 0.1)
        """
        super(Transformer, self).__init__()

        # Instantiate the Encoder layer
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)

        # Instantiate the Decoder layer
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)

        # Final Dense layer to generate vocabulary probabilities
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Calculates the output of the transformer network.

        Args:
            inputs: a tensor of shape (batch, input_seq_len) containing
                    the inputs
            target: a tensor of shape (batch, target_seq_len) containing
                    the target
            training: a boolean to determine if the model is training
            encoder_mask: the padding mask to be applied to the encoder
            look_ahead_mask: the look ahead mask to be applied to the decoder
            decoder_mask: the padding mask to be applied to the decoder

        Returns:
            A tensor of shape (batch, target_seq_len, target_vocab) containing
            the transformer output.
        """
        # Pass inputs through the encoder
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Pass targets and encoder output through the decoder
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # Pass the decoder output through the final linear layer
        final_output = self.linear(dec_output)

        return final_output

#!/usr/bin/env python3
"""RNN Decoder module for machine translation"""
import tensorflow as tf


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class RNNDecoder that inherits from tensorflow.keras.layers.Layer
    to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        Args:
            vocab: integer representing the size of the output vocabulary
            embedding: integer representing the dimensionality of the
                       embedding vector
            units: integer representing the number of hidden units in the
                   RNN cell
            batch: integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Public instance method to perform the forward pass
        Args:
            x: tensor of shape (batch, 1) containing the previous word in
               the target sequence as an index of the target vocabulary
            s_prev: tensor of shape (batch, units) containing the previous
                    decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                           containing the outputs of the encoder
        Returns:
            y: tensor of shape (batch, vocab) containing the output word
               as a one hot vector in the target vocabulary
            s: tensor of shape (batch, units) containing the new decoder
               hidden state
        """
        SelfAttention = __import__('1-self_attention').SelfAttention

        attention = SelfAttention(self.gru.units)
        context, _ = attention(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        output, s = self.gru(x)
        output = tf.squeeze(output, axis=1)
        y = self.F(output)

        return y, s

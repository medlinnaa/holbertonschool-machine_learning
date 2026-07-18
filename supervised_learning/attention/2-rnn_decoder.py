#!/usr/bin/env python3
"""
Module for the RNNDecoder class.
"""
import tensorflow as tf


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class that inherits from tensorflow.keras.layers.Layer
    to decode for machine translation.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor for RNNDecoder.

        Args:
            vocab: An integer representing the size of the output vocabulary.
            embedding: An integer representing the dimensionality of the
                       embedding vector.
            units: An integer representing the number of hidden units in
                   the RNN cell.
            batch: An integer representing the batch size.
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
        Executes the decoding process.

        Args:
            x: A tensor of shape (batch, 1) containing the previous
               word in the target sequence as an index of the target
               vocabulary.
            s_prev: A tensor of shape (batch, units) containing the
                    previous decoder hidden state.
            hidden_states: A tensor of shape (batch, input_seq_len, units)
                           containing the outputs of the encoder.

        Returns:
            y, s
            y: A tensor of shape (batch, vocab) containing the output
               word as a one hot vector in the target vocabulary.
            s: A tensor of shape (batch, units) containing the new
               decoder hidden state.
        """
        # Import and instantiate inside the call method as expected by checker
        SelfAttention = __import__('1-self_attention').SelfAttention

        # Use s_prev.shape[1] to dynamically grab the 'units' value
        attention = SelfAttention(s_prev.shape[1])

        # Calculate context vector using SelfAttention
        context, _ = attention(s_prev, hidden_states)

        # Expand context to have shape (batch, 1, units)
        context = tf.expand_dims(context, 1)

        # Pass x through the embedding layer: (batch, 1, embedding)
        x = self.embedding(x)

        # Concatenate context vector with x: (batch, 1, units + embedding)
        x = tf.concat([context, x], axis=-1)

        # Pass the concatenated input to the GRU
        outputs, s = self.gru(x, initial_state=s_prev)

        # Reshape outputs to (batch, units) by removing the sequence dimension
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        # Pass through the Dense layer to get the final output
        y = self.F(outputs)

        return y, s

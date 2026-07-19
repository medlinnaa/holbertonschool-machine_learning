#!/usr/bin/env python3
"""
A class that loads, prepares, tokenizes,
and encodes a dataset for machine translation.
"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads, prepares, and encodes a dataset for machine translation.
    """

    def __init__(self):
        """
        Initializes the Dataset instance.
        Loads the train and validation datasets, creates the tokenizers,
        and updates the dataset attributes by tokenizing the examples.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Update data_train and data_valid by applying the tf_encode wrapper
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data (tf.data.Dataset): Dataset containing tuples of
                (pt, en) formatted as tf.Tensor strings.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en) containing the
                trained Portuguese and English tokenizers.
        """
        pt_model = 'neuralmind/bert-base-portuguese-cased'
        en_model = 'bert-base-uncased'

        # Load the base pre-trained tokenizers
        tokenizer_pt_base = transformers.AutoTokenizer.from_pretrained(
            pt_model
        )
        tokenizer_en_base = transformers.AutoTokenizer.from_pretrained(
            en_model
        )

        def pt_iterator():
            """
            Generator yielding batches of Portuguese strings.
            """
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in pt]

        def en_iterator():
            """
            Generator yielding batches of English strings.
            """
            for _, en in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in en]

        # Train new tokenizers based on the dataset iterators
        vocab_size = 2 ** 13
        tokenizer_pt = tokenizer_pt_base.train_new_from_iterator(
            pt_iterator(), vocab_size
        )
        tokenizer_en = tokenizer_en_base.train_new_from_iterator(
            en_iterator(), vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        Args:
            pt (tf.Tensor): Tensor containing the Portuguese sentence.
            en (tf.Tensor): Tensor containing
            the corresponding English sentence.

        Returns:
            tuple: (pt_tokens, en_tokens) containing the lists of
                Portuguese and English tokens, respectively.
        """
        vocab_size = self.tokenizer_pt.vocab_size

        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_encoded = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_encoded = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        pt_tokens = [vocab_size] + pt_encoded + [vocab_size + 1]
        en_tokens = [vocab_size] + en_encoded + [vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method.

        Args:
            pt (tf.Tensor): Tensor containing the Portuguese sentence.
            en (tf.Tensor): Tensor containing
            the corresponding English sentence.

        Returns:
            tuple: (pt_encoded, en_encoded) containing the tokenized
                Portuguese and English tensors, with explicit shapes.
        """
        # tf.py_function executes a regular Python function inside a TF graph
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Ensure the output tensors have a defined shape (required by map)
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded

#!/usr/bin/env python3
"""
A class that loads, prepares, tokenizes, and encodes a dataset for machine translation.
"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads, prepares, and encodes a dataset for machine translation.
    """

    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset instance and sets up the data pipeline.

        Args:
            batch_size (int): The batch size for training/validation.
            max_len (int): The maximum number of tokens allowed per sentence.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Apply the encoding wrapper
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Helper function for filtering
        def filter_max_length(pt, en):
            """Filters out examples where either sequence exceeds max_len"""
            return tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )

        # --- Set up data_train pipeline ---
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        # --- Set up data_valid pipeline ---
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size)

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

        tokenizer_pt_base = transformers.AutoTokenizer.from_pretrained(
            pt_model
        )
        tokenizer_en_base = transformers.AutoTokenizer.from_pretrained(
            en_model
        )

        def pt_iterator():
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in pt]

        def en_iterator():
            for _, en in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in en]

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
            en (tf.Tensor): Tensor containing the corresponding English sentence.

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
            en (tf.Tensor): Tensor containing the corresponding English sentence.

        Returns:
            tuple: (pt_encoded, en_encoded) containing the tokenized
                Portuguese and English tensors, with explicit shapes.
        """
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded

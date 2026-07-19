#!/usr/bin/env python3
"""
Dataset module for loading and tokenizing a translation dataset.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and prepares a dataset for machine translation.
    """

    def __init__(self):
        """
        Initializes the Dataset instance.
        Loads the train and validation datasets and creates the tokenizers.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

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
            Using batching speeds up the tokenizer training process.
            """
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in pt]

        def en_iterator():
            """
            Generator yielding batches of English strings.
            Using batching speeds up the tokenizer training process.
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

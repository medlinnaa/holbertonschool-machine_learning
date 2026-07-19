#!/usr/bin/env python3
"""
Dataset module for loading, tokenizing, and encoding a translation dataset.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads, prepares, and encodes a dataset for machine translation.
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
            en (tf.Tensor): Tensor containing the corresponding English sentence.

        Returns:
            tuple: (pt_tokens, en_tokens) containing the lists of
                Portuguese and English tokens, respectively.
        """
        # Define the target vocabulary size (used for start/end tokens)
        vocab_size = self.tokenizer_pt.vocab_size

        # Extract the strings from the tensors
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Encode without adding default Hugging Face special tokens (like CLS)
        pt_encoded = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_encoded = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        # Wrap with start token (vocab_size) and end token (vocab_size + 1)
        pt_tokens = [vocab_size] + pt_encoded + [vocab_size + 1]
        en_tokens = [vocab_size] + en_encoded + [vocab_size + 1]

        return pt_tokens, en_tokens

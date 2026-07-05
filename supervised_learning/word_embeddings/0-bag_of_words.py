#!/usr/bin/env python3
"""
Module for extracting a Bag of Words embedding matrix.
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list): List of vocabulary words to use for the analysis.
            If None, all words within sentences should be used.

    Returns:
        tuple: (embeddings, features)
            embeddings is a numpy.ndarray of shape (s, f) containing counts.
            features is a numpy.ndarray of the features used for embeddings.
    """
    processed_docs = []
    unique_words = set()

    # Pre-process strings to clean text and extract words
    for sentence in sentences:
        s = sentence.lower()
        # Remove possessive 's (e.g. "children's" -> "children")
        s = re.sub(r"'s\b", "", s)
        # Extract alphanumeric words using regex word boundaries
        words = re.findall(r"\b\w+\b", s)
        processed_docs.append(words)

        if vocab is None:
            unique_words.update(words)

    # Establish the feature vocabulary
    if vocab is None:
        features = sorted(list(unique_words))
    else:
        features = vocab

    s_len = len(sentences)
    f_len = len(features)

    # Initialize the embedding matrix with zeros
    embeddings = np.zeros((s_len, f_len), dtype=int)

    # Populate the embedding matrix
    for i, words in enumerate(processed_docs):
        for word in words:
            if word in features:
                j = features.index(word)
                embeddings[i, j] += 1

    return embeddings, np.array(features)

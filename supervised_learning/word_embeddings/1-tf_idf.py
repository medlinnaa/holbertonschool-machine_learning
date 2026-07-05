#!/usr/bin/env python3
"""
Module for extracting a TF-IDF embedding matrix.
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list): List of vocabulary words to use for the analysis.
            If None, all words within sentences should be used.

    Returns:
        tuple: (embeddings, features)
            embeddings is a numpy.ndarray of shape (s, f) containing the
                TF-IDF scores.
            features is a numpy.ndarray of the features used for embeddings.
    """
    # Initialize the vectorizer with the specific vocabulary if provided
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences into the TF-IDF matrix
    X = vectorizer.fit_transform(sentences)

    # Extract the feature names (vocabulary) used
    features = vectorizer.get_feature_names_out()

    # Convert the sparse matrix X to a dense numpy array
    return X.toarray(), features

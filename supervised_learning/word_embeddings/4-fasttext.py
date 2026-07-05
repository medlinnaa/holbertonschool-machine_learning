#!/usr/bin/env python3
"""
Module to create, build, and train a fastText model using Gensim.
"""
from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim fastText model.

    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum occurrences of a word for use in training.
        negative (int): Size of negative sampling.
        window (int): Max distance between current and predicted word.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        The trained FastText model.
    """
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model

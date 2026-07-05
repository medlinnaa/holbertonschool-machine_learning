#!/usr/bin/env python3
"""
Module to create, build, and train a word2vec model using Gensim.
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                    negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model.
    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum occurrences of a word for use in training.
        window (int): Max distance between current and predicted word.
        negative (int): Size of negative sampling.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.
    Returns:
        The trained Word2Vec model.
    """
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model

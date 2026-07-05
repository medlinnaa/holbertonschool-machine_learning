#!/usr/bin/env python3
"""
Module to create, build, and train a word2vec model using Gensim.
"""
import gensim


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model.
    """
    # Check if the environment is running Gensim 4.0 or newer
    if int(gensim.__version__[0]) >= 4:
        model = gensim.models.Word2Vec(
            sentences=sentences,
            vector_size=size,      # Mapped from 'size'
            min_count=min_count,
            window=window,
            negative=negative,
            sg=0 if cbow else 1,
            epochs=iterations,     # Mapped from 'iterations'
            seed=seed,
            workers=workers
        )
    else:
        model = gensim.models.Word2Vec(
            sentences=sentences,
            size=size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=0 if cbow else 1,
            iter=iterations,
            seed=seed,
            workers=workers
        )
    
    return model

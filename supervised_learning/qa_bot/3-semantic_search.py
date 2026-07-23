#!/usr/bin/env python3
"""Module for Question Answering using a pre-trained BERT model."""

import os
import tensorflow_hub as hub
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents to find the most similar text.

    Args:
        corpus_path (str): The path to the corpus of reference documents.
        sentence (str): The sentence from which to perform semantic search.

    Returns:
        str: The reference text of the document most similar to the sentence.
    """
    # Load the Universal Sentence Encoder model
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    documents = []

    # Read all markdown files from the provided path
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())

    # Create a list with the sentence at the 0th index, followed by all documents
    corpus = [sentence] + documents

    # Generate embeddings for the sentence and all documents
    embeddings = model(corpus)

    # Extract the sentence embedding and the document embeddings
    sentence_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    # Compute the correlation (cosine similarity) between the sentence and documents
    correlations = np.inner(sentence_embedding, doc_embeddings)

    # Find the index of the document with the highest similarity score
    most_similar_idx = np.argmax(correlations)

    # Return the text of the most similar document
    return documents[most_similar_idx]

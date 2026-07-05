#!/usr/bin/env python3
"""
Module to calculate the n-gram BLEU score for a sentence.
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list): List of reference translations, where each
            reference translation is a list of the words.
        sentence (list): List containing the model proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: The n-gram BLEU score.
    """
    c = len(sentence)

    # Early exit if the proposed sentence is too short to form an n-gram
    if c < n:
        return 0.0

    # 1. Calculate Brevity Penalty (BP)
    ref_lengths = [len(ref) for ref in references]
    # Find closest reference length, defaulting to shortest if there is a tie
    r = min(ref_lengths, key=lambda x: (abs(x - c), x))

    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - (r / c))

    # 2. Extract n-grams for the candidate sentence
    cand_ngrams = []
    for i in range(c - n + 1):
        cand_ngrams.append(tuple(sentence[i:i + n]))

    unique_cand_ngrams = set(cand_ngrams)

    # 3. Extract n-grams for all reference sentences
    ref_ngrams_list = []
    for ref in references:
        r_ngrams = []
        for i in range(len(ref) - n + 1):
            r_ngrams.append(tuple(ref[i:i + n]))
        ref_ngrams_list.append(r_ngrams)

    # 4. Calculate N-gram Precision (p_n) with clipping
    matches = 0
    for ngram in unique_cand_ngrams:
        # Count occurrences in the candidate sentence
        count_candidate = cand_ngrams.count(ngram)
        # Find the maximum occurrences of this n-gram in any single reference
        max_ref_count = max(
            [r_ngrams.count(ngram) for r_ngrams in ref_ngrams_list]
        )
        # Add the clipped count
        matches += min(count_candidate, max_ref_count)

    total_cand_ngrams = len(cand_ngrams)
    p_n = matches / total_cand_ngrams

    # Return the final isolated n-gram BLEU score
    return bp * p_n

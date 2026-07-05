#!/usr/bin/env python3
"""
Module to calculate the cumulative n-gram BLEU score for a sentence.
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): List of reference translations, where each
            reference translation is a list of the words.
        sentence (list): List containing the model proposed sentence.
        n (int): The size of the largest n-gram to use for evaluation.

    Returns:
        float: The cumulative n-gram BLEU score.
    """
    c = len(sentence)
    if c == 0:
        return 0.0

    # 1. Calculate Brevity Penalty (BP)
    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda x: (abs(x - c), x))

    bp = 1.0 if c > r else np.exp(1 - (r / c))

    precisions = []

    # 2. Iterate from 1-gram up to n-gram
    for i in range(1, n + 1):
        cand_ngrams = [
            tuple(sentence[k:k + i]) for k in range(c - i + 1)
        ]

        # If the candidate sentence is too short for this n-gram size
        if not cand_ngrams:
            precisions.append(0.0)
            continue

        unique_cand = set(cand_ngrams)

        ref_ngrams_list = [
            [tuple(ref[k:k + i]) for k in range(len(ref) - i + 1)]
            for ref in references
        ]

        matches = 0
        for ngram in unique_cand:
            count_cand = cand_ngrams.count(ngram)
            max_ref_count = max(
                r_ngrams.count(ngram) for r_ngrams in ref_ngrams_list
            )
            matches += min(count_cand, max_ref_count)

        p_i = matches / len(cand_ngrams)
        precisions.append(p_i)

    # 3. Calculate the Geometric Mean of the precisions
    # If any precision is 0, the entire geometric mean (and BLEU) becomes 0
    if 0.0 in precisions:
        return 0.0

    # Using numpy to calculate the weighted sum of logs evenly (1/n)
    geom_mean = np.exp(np.sum((1 / n) * np.log(precisions)))

    # Return the final cumulative BLEU score
    return bp * geom_mean

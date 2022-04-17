"""
Functions for measuring cost between two Sequence objects
"""

import numpy as np
from Bio.SubsMat import MatrixInfo

from mgm.data.kuzmin_data import aa_vocab


def squared_difference(seq1, seq2, representation = 'one-hot'):
    assert (len(seq1) == len(seq2))
    x1 = seq1.get_encoding(representation)
    x2 = seq2.get_encoding(representation)
    return 0.5 * np.linalg.norm(x1-x2, ord='fro')**2

def num_differences(seq1, seq2):
    assert(len(seq1) == len(seq2))
    array1 = np.array(seq1.integer_encoded)
    array2 = np.array(seq2.integer_encoded)
    return np.count_nonzero(array1 != array2)

"""
Code for similarity scoring of aligned sequences
"""

def score_match(pair, matrix):
    if pair in matrix:
        return matrix[pair]
    elif (tuple(reversed(pair))) in matrix:
        return matrix[(tuple(reversed(pair)))]
    else:
        return 0

def score_pairwise(char_seq1, char_seq2, matrix=MatrixInfo.blosum62, gap_s=-2, gap_e=-1):
    """
    Takes two sequences of aa characters, already aligned. Returns the similarity score.
    """
    score = 0
    gap = False
    for i in range(len(char_seq1)):
        pair = (char_seq1[i], char_seq2[i])
        if pair == ('-', '-'):  # Ignore two aligned gaps
            continue

        if not gap:
            if '-' in pair:
                gap = True
                score += gap_s
            else:
                score += score_match(pair, matrix)
        else:
            if '-' not in pair:
                gap = False
                score += score_match(pair, matrix)
            else:
                score += gap_e
    return score

def similarity(seq1, seq2):
    """
    Compute the Blossom similarity score of two Sequence objects.
    """
    str1 = ''.join([aa_vocab[i] for i in seq1.integer_encoded])
    str2 = ''.join([aa_vocab[i] for i in seq2.integer_encoded])
    return score_pairwise(str1, str2)
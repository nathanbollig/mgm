"""
Functions for measuring cost between two Sequence objects
"""

import numpy as np

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


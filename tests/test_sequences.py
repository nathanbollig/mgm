"""
Test the Sequence class
"""

from mgm.common.sequence import Sequence, mult_align_idx_to_unaligned_idx, unaligned_idx_to_mult_align_idx
import pytest
import numpy as np
import pickle

def test_sequences():
    # Test constructor argument verification
    x = [1,2,0,3,4,1,2]
    aa_vocab = ['a','b','c','d','e']

    with pytest.raises(ValueError):
        Sequence(x, n_characters=6, aa_vocab=aa_vocab)

    seq = Sequence(x, n_characters=5, aa_vocab=aa_vocab)
    assert(seq.n_positions == 7)

    seq = Sequence(x, aa_vocab=aa_vocab)
    assert(seq.n_positions == 7)
    assert(seq.n_characters == 5)

    seq = Sequence(x, n_characters=5)

    with pytest.raises(ValueError):
        seq = Sequence(x)
    with pytest.raises(ValueError):
        seq = Sequence(x, n_positions=9, n_characters=5, aa_vocab=aa_vocab)
    with pytest.raises(ValueError):
        seq = Sequence(x, n_positions=6, aa_vocab=aa_vocab)
    with pytest.raises(ValueError):
        seq = Sequence(x, n_positions=5, n_characters=5)

    seq1 = Sequence(x, n_positions=7, n_characters=5, aa_vocab=aa_vocab)
    seq2 = Sequence(x, n_positions=7, n_characters=5)
    seq3 = Sequence(x, n_positions=7, aa_vocab=aa_vocab)

    assert(np.all(seq1.integer_encoded == seq2.integer_encoded))
    assert(np.all(seq2.integer_encoded == seq3.integer_encoded))
    assert(np.all(seq1.one_hot_encoded == seq2.one_hot_encoded))
    assert(np.all(seq2.one_hot_encoded == seq3.one_hot_encoded))

    # Case: x is an integer-encoded list
    assert(np.all(seq1.integer_encoded == x))
    expected_x = np.array([[0,1,0,0,0], [0,0,1,0,0], [1,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,1,0,0,0], [0,0,1,0,0]])
    assert(np.all(seq.one_hot_encoded == expected_x))

    # Case: x is a 1d array
    x_int = x
    x = np.array(x)
    seq4 = Sequence(x, n_positions=7, aa_vocab=aa_vocab)
    assert(np.all(seq1.integer_encoded == seq4.integer_encoded))
    assert(np.all(seq1.one_hot_encoded == seq4.one_hot_encoded))

    # Case: x is a 2d array
    x = expected_x

    with pytest.raises(ValueError):
        seq = Sequence(x, n_positions=9, n_characters=5, aa_vocab=aa_vocab)
    with pytest.raises(ValueError):
        seq = Sequence(x, n_positions=6, aa_vocab=aa_vocab)
    with pytest.raises(ValueError):
        seq = Sequence(x, n_positions=5, n_characters=5)

    seq = Sequence(x, n_positions=7, aa_vocab=aa_vocab)
    assert(np.all(seq.integer_encoded == seq4.integer_encoded))
    assert(np.all(seq.one_hot_encoded == seq4.one_hot_encoded))



    # Try passing a generated sequence into a Sequence object
    with open("sample_data/data_test.pkl", 'rb') as pfile:
        data_test = pickle.load(pfile)

    X, y = data_test

    seq = Sequence(X[0], y[0], n_characters = 20)
    assert(seq.n_positions == 60)
    assert(seq.y == y[0])
    assert(np.all(seq.one_hot_encoded == X[0]))
    assert(seq.integer_encoded[0] == np.argmax(X[0], axis=1)[0])

    # Test to_predict
    assert(np.all(seq.to_predict()[0] == X[0]))

    # Test length
    assert(len(seq) == 60)

    # Test substitution
    x = [1,2,0,3,4,1,2]
    aa_vocab = ['a','b','c','d','e']
    seq = Sequence(x, n_positions=7, aa_vocab=aa_vocab)
    assert(seq.integer_encoded[2] == 0)
    assert(np.all(seq.one_hot_encoded[2] == np.array([1,0,0,0,0])))
    seq.sub(2, 1)
    assert(seq.integer_encoded[2] == 1)
    assert(np.all(seq.one_hot_encoded[2] == np.array([0,1,0,0,0])))

    # Test copy
    seq_copy = seq.copy()
    assert(seq.integer_encoded[2] == 1)
    assert(seq_copy.integer_encoded[2] == 1)
    seq_copy.sub(2,4)
    assert(seq.integer_encoded[2] == 1)
    assert(seq_copy.integer_encoded[2] == 4)

    # Test sub with gap
    seq.sub(0,-1)
    assert(seq.integer_encoded[0] == -1)
    assert(np.all(seq.one_hot_encoded[0] == np.array([0,0,0,0,0])))

    # Test Kidera representation space
    seq.sub(0, 0)
    assert(seq.aa_vocab[0] == 'A')
    R_kidera = seq.representation_space['kidera']
    assert(R_kidera.shape == (seq.n_characters, 10))

    # Test get_encoding - one-hot
    oh1 = seq.get_encoding('one-hot')
    oh2 = seq.one_hot_encoded
    assert(np.all(oh1==oh2))

    # Test get_encoding - kidera
    k = seq.get_encoding('kidera')
    i = 1
    char = seq.integer_encoded[i]
    rep = R_kidera[char, :]
    assert(np.all(np.isclose(rep, k[i, :])))

    # Test index conversion methods
    seq1 = [1, 2, -1, -1, -1, 3, 4, -1, 5, -1, -1, -1, -1, -1]

    assert (mult_align_idx_to_unaligned_idx(seq1, 0) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq1, 1) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq1, 2) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq1, 3) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq1, 4) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq1, 5) == 2)
    assert (mult_align_idx_to_unaligned_idx(seq1, 6) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq1, 7) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq1, 8) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq1, 9) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq1, 10) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq1, 11) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq1, 12) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq1, 13) == 4)
    with pytest.raises(IndexError):
        mult_align_idx_to_unaligned_idx(seq1, 14)

    assert (unaligned_idx_to_mult_align_idx(seq1, 0) == 0)
    assert (unaligned_idx_to_mult_align_idx(seq1, 1) == 1)
    assert (unaligned_idx_to_mult_align_idx(seq1, 2) == 5)
    assert (unaligned_idx_to_mult_align_idx(seq1, 3) == 6)
    assert (unaligned_idx_to_mult_align_idx(seq1, 4) == 8)
    with pytest.raises(IndexError):
        unaligned_idx_to_mult_align_idx(seq1, 5)

    seq2 = [1, 2, -1, -1, -1, 3, 4, -1, 5, -1, -1, -1, -1, 6]

    assert (mult_align_idx_to_unaligned_idx(seq2, 0) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq2, 1) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq2, 2) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq2, 3) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq2, 4) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq2, 5) == 2)
    assert (mult_align_idx_to_unaligned_idx(seq2, 6) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq2, 7) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq2, 8) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq2, 9) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq2, 10) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq2, 11) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq2, 12) == 4)
    assert (mult_align_idx_to_unaligned_idx(seq2, 13) == 5)
    with pytest.raises(IndexError):
        mult_align_idx_to_unaligned_idx(seq2, 14)

    assert (unaligned_idx_to_mult_align_idx(seq2, 0) == 0)
    assert (unaligned_idx_to_mult_align_idx(seq2, 1) == 1)
    assert (unaligned_idx_to_mult_align_idx(seq2, 2) == 5)
    assert (unaligned_idx_to_mult_align_idx(seq2, 3) == 6)
    assert (unaligned_idx_to_mult_align_idx(seq2, 4) == 8)
    assert (unaligned_idx_to_mult_align_idx(seq2, 5) == 13)
    with pytest.raises(IndexError):
        unaligned_idx_to_mult_align_idx(seq2, 6)

    seq3 = [-1, 2, -1, -1, -1, 3, 4, -1, 5, -1, -1, -1, -1, -1]

    assert (mult_align_idx_to_unaligned_idx(seq3, 0) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq3, 1) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq3, 2) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq3, 3) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq3, 4) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq3, 5) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq3, 6) == 2)
    assert (mult_align_idx_to_unaligned_idx(seq3, 7) == 2)
    assert (mult_align_idx_to_unaligned_idx(seq3, 8) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq3, 9) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq3, 10) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq3, 11) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq3, 12) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq3, 13) == 3)
    with pytest.raises(IndexError):
        mult_align_idx_to_unaligned_idx(seq3, 14)

    assert (unaligned_idx_to_mult_align_idx(seq3, 0) == 1)
    assert (unaligned_idx_to_mult_align_idx(seq3, 1) == 5)
    assert (unaligned_idx_to_mult_align_idx(seq3, 2) == 6)
    assert (unaligned_idx_to_mult_align_idx(seq3, 3) == 8)
    with pytest.raises(IndexError):
        unaligned_idx_to_mult_align_idx(seq3, 4)

    seq4 = [-1, 2, -1, -1, -1, 3, 4, -1, 5, -1, -1, -1, -1, 6]

    assert (mult_align_idx_to_unaligned_idx(seq4, 0) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq4, 1) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq4, 2) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq4, 3) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq4, 4) == 0)
    assert (mult_align_idx_to_unaligned_idx(seq4, 5) == 1)
    assert (mult_align_idx_to_unaligned_idx(seq4, 6) == 2)
    assert (mult_align_idx_to_unaligned_idx(seq4, 7) == 2)
    assert (mult_align_idx_to_unaligned_idx(seq4, 8) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq4, 9) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq4, 10) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq4, 11) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq4, 12) == 3)
    assert (mult_align_idx_to_unaligned_idx(seq4, 13) == 4)
    with pytest.raises(IndexError):
        mult_align_idx_to_unaligned_idx(seq4, 14)

    assert (unaligned_idx_to_mult_align_idx(seq4, 0) == 1)
    assert (unaligned_idx_to_mult_align_idx(seq4, 1) == 5)
    assert (unaligned_idx_to_mult_align_idx(seq4, 2) == 6)
    assert (unaligned_idx_to_mult_align_idx(seq4, 3) == 8)
    assert (unaligned_idx_to_mult_align_idx(seq4, 4) == 13)
    with pytest.raises(IndexError):
        unaligned_idx_to_mult_align_idx(seq4, 5)

    print("Passed!")

if __name__ == "__main__":
    test_sequences()
"""
Test the Sequence class
"""

from mgm.common.sequence import Sequence
import pytest
import numpy as np
import pickle

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

assert(seq1.integer_encoded == seq2.integer_encoded)
assert(seq2.integer_encoded == seq3.integer_encoded)
assert(np.all(seq1.one_hot_encoded == seq2.one_hot_encoded))
assert(np.all(seq2.one_hot_encoded == seq3.one_hot_encoded))

# Case: x is an integer-encoded list
assert(seq1.integer_encoded == x)
expected_x = np.array([[0,1,0,0,0], [0,0,1,0,0], [1,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,1,0,0,0], [0,0,1,0,0]])
assert(np.all(seq.one_hot_encoded == expected_x))

# Case: x is a 1d array
x_int = x
x = np.array(x)
seq4 = Sequence(x, n_positions=7, aa_vocab=aa_vocab)
assert(seq1.integer_encoded == seq4.integer_encoded)
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
assert(seq.integer_encoded == seq4.integer_encoded)
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


print("Passed!")
from mgm.common.sequence import Sequence
from mgm.common.cost_functions import squared_difference, num_differences
import pickle
import numpy as np

def test_costs():
    with open("sample_data/data_test.pkl", 'rb') as pfile:
        data_test = pickle.load(pfile)

    X, y = data_test

    seq1 = Sequence(X[0], y[0], n_characters = 20)
    seq2 = Sequence(X[1], y[1], n_characters = 20)

    cost = squared_difference(seq1, seq2)

    cost_alt = num_differences(seq1, seq2)

    assert(np.isclose(cost, cost_alt))

print("Passed!")

if __name__ == "__main__":
    test_costs()
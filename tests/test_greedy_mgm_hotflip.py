"""
Test single application of hotflip using the greedy mgm wrapper.
Data is read one-hot encoded but passed to the wrapper as a Sequence.

pd.DataFrame(data) is saved in test directory as sample_hotflip_data.csv
"""

import pickle
import keras
from mgm.algorithms.mutations import variant_search
from mgm.common.sequence import Sequence
import pandas as pd
import numpy as np
from mgm.models.NN import make_LSTM


def test_greedy_mgm_hotflip():
    # Load model
    model = keras.models.load_model('sample_data/model.tf')

    # Load data
    with open("sample_data/data_test.pkl", 'rb') as pfile:
        data_test = pickle.load(pfile)

    X, y = data_test

    # Load aa vocab
    with open("sample_data/aa_vocab.pkl", 'rb') as pfile:
        aa_vocab = pickle.load(pfile)

    # Find first negative instance in the dataset
    for i in range(X.shape[0]):
        if y[i] == 0:
            break
        else:
            i += 1

    # Create a Sequence object
    x = X[i]
    seq = Sequence(x, y[i], aa_vocab)

    # Adjust first character to gap
    seq.sub(0,-1)

    # Apply hotflip using the greedy mgm wrapper
    hx = variant_search(seq, model=model, confidence_threshold = 0.9, type="hotflip", verbose=True)
    data = hx.substitution_data

    # Tests on data
    pd.DataFrame(data)
    assert(len(data) > 0)
    assert(len(data) == 3)
    assert(data[0]['pos_to_change'] == 29)
    assert(data[0]['new_char_idx'] == 9)
    assert(data[1]['pos_to_change'] == 33)
    assert(data[1]['new_char_idx'] == 10)
    assert(data[2]['pos_to_change'] == 32)
    assert(data[2]['new_char_idx'] == 10)
    assert(data[2]['change_number'] == 3)
    assert(data[2]['time_sec'] < 0.1)

    # Additional test using gradient wtr model conf instead of loss
    hx2 = variant_search(seq, model=model, confidence_threshold = 0.9, type="hotflip", verbose=True, loss=False)
    data = hx2.substitution_data
    pd.DataFrame(data)
    assert (len(data) > 0)
    assert (len(data) == 3)
    assert (data[0]['pos_to_change'] == 29)
    assert (data[0]['new_char_idx'] == 9)
    assert (data[1]['pos_to_change'] == 33)
    assert (data[1]['new_char_idx'] == 10)
    assert (data[2]['pos_to_change'] == 32)
    assert (data[2]['new_char_idx'] == 10)
    assert (data[2]['change_number'] == 3)
    assert (data[2]['time_sec'] < 0.2)

    print("Passed!")

if __name__ == "__main__":
    test_greedy_mgm_hotflip()
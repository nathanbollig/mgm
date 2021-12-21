

"""
Pipelines for Kuzmin data
"""
from mgm.algorithms.mutations import greedy_mgm
from mgm.analysis.risk_assessment import risk_of_variant
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import load_kuzmin_data
from mgm.models.NN import make_LSTM
import pickle
import numpy as np

def exp1():
    # Load Data
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
    n_positions = X.shape[1]
    n_characters = X.shape[2]
    assert(n_positions == 2396)

    # Find first negative instance in the dataset
    for i in range(X.shape[0]):
        if y[i] == 0:
            break
        else:
            i += 1

    # Train Model
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    model = make_LSTM(X_train, y_train, N_POS=n_positions)
    model.fit(X_train, y_train, epochs=10)

    # Create a Sequence object
    x = X[i]
    seq = Sequence(x, y[i], n_characters=n_characters, n_positions=n_positions)

    # Run MGM
    variant = greedy_mgm(seq, model=model, confidence_threshold = 0.9, type="hotflip", verbose=True)

    # Get risk score
    risk_score = risk_of_variant(variant)

    assert(1==0)

if __name__ == "__main__":
    exp1()

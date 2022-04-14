from mgm.algorithms.mutations import variant_search
from mgm.common.utils import set_data_directory
from mgm.data.kuzmin_data import load_kuzmin_data
from mgm.algorithms.mgm import mgm_d
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import aa_vocab
from mgm.models.NN import make_LSTM, make_CNN
import numpy as np

def test_kidera_mgm_d():
    # Load data with kidera representation
    X, y, species, deflines, sequences, sp, human_virus_species_list, seqs = load_kuzmin_data(representation_type='kidera')

    # Train model on kidera representation
    n_positions = X.shape[1]
    n_characters = 25
    assert (n_positions == 2396)

    # Find first negative instance in the dataset
    for i in range(1, X.shape[0]):
        if y[i] == 0:
            break
        else:
            i += 1

    # Train kidera model
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    model = make_LSTM(X_train, y_train, N_POS=n_positions)
    model.fit(X_train, y_train, epochs=2)

    # Retreive Sequence object
    seq = seqs[i]

    # Train one-hot model
    seqs_train = seqs[:i] + seqs[i+1 :]
    X_train_oh = []
    for s in seqs_train:
        X_train_oh.append(s.get_encoding('one-hot'))
    X_train_oh = np.array(X_train_oh)
    model_oh = make_LSTM(X_train_oh, y_train, N_POS=n_positions)
    model_oh.fit(X_train_oh, y_train, epochs=2)

    # Run mgm-d with kidera-encoded sequences and corresponding model
    seq_kid, data_kid = mgm_d(seq.copy(), seq, model=model, representation='kidera', cost_function='squared_difference')
    seq_oh, data_oh = mgm_d(seq.copy(), seq, model=model_oh, representation='one-hot', cost_function='squared_difference')

    # Test LOOCV
    # model_initializer = make_CNN
    # LOOCV(model_initializer, X, y, species, epochs=5, output_string="test")

    # Run variant searches
    hx_hf = variant_search(seq, model=model_oh, confidence_threshold=0.9, type="hotflip", verbose=True, fixed_iterations=3)
    hx_mgm_oh = variant_search(seq, model=model_oh, confidence_threshold=0.9, type="mgm-d", verbose=True, fixed_iterations=3)
    hx_mgm_kid = variant_search(seq, model=model, confidence_threshold=0.9, type="mgm-d", representation='kidera', verbose=True, fixed_iterations=3)



    assert(1==0)

    print("Passed!")

if __name__ == "__main__":
    set_data_directory("test_kidera3")
    test_kidera_mgm_d()
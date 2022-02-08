from mgm.common.utils import set_data_directory
from mgm.data.kuzmin_data import load_kuzmin_data
from mgm.algorithms.mgm import mgm_d
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import aa_vocab
from mgm.models.NN import make_LSTM, make_CNN
import numpy as np

def test_kidera_mgm_d():
    # Load data with kidera representation
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data(representation_type='kidera')

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

    # Train Model
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    model = make_LSTM(X_train, y_train, N_POS=n_positions)
    model.fit(X_train, y_train, epochs=2)

    # Create a Sequence object
    x = X[i]
    seq = Sequence(x, y[i], n_characters=n_characters, n_positions=n_positions)

    # Run mgm-d with kidera-encoded sequences and corresponding model
    seq_mgm_d, data_mgm_d = mgm_d(seq.copy(), seq, model=model, representation='kidera', cost_function='squared_difference', lambda_param=1)


    # Test LOOCV
    # model_initializer = make_CNN
    # LOOCV(model_initializer, X, y, species, epochs=5, output_string="test")

    assert(1==0)

    print("Passed!")

if __name__ == "__main__":
    set_data_directory("test_kidera")
    test_kidera_mgm_d()
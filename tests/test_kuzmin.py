"""
Use load_kuzmin_data to retrieve data objects, and trains a LSTM using the species-aware 7-fold CV split.
Training only for 1 epoch with large batch size to increase training speed.

Takes ~ 7 min
"""
from mgm.common.utils import set_data_directory
from mgm.data.kuzmin_data import load_kuzmin_data, species_aware_CV, LOOCV
from mgm.models.NN import make_LSTM, make_CNN
import numpy as np



def test_kuzmin():
    # Run data load
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()

    # Test sp structure
    n = 0
    for i in range(len(human_virus_species_list)):
        n += len(sp[i])
    assert(n == np.sum(y))
    assert(len(sp['non-human']) == (len(y) - n))
    assert(np.sum(y[sp['non-human']]) == 0)
    for i in range(len(human_virus_species_list)):
        assert(np.all(y[sp[i]]==1)==True)

    model_initializer = make_CNN
    #species_aware_CV(model_initializer, X, y, species, human_virus_species_list, epochs=10, output_string="test00", remove_duplicate_species=True)
    LOOCV(model_initializer, X, y, species, epochs=5, output_string="test")

if __name__ == "__main__":
    set_data_directory("test_kuzmin4")
    test_kuzmin()
"""
Use load_kuzmin_data to retrieve data objects, and trains a LSTM using the species-aware 7-fold CV split.
Training only for 1 epoch with large batch size to increase training speed.

Takes ~ 7 min
"""
import datetime

from mgm.common.utils import set_data_directory
from mgm.data.kuzmin_data import load_kuzmin_data, species_aware_CV, LOOCV
from mgm.models.NN import make_LSTM, make_CNN
import numpy as np



def test_kuzmin():
    # Run data load
    X, y, species, deflines, sequences, sp, human_virus_species_list, seqs = load_kuzmin_data()

    # Test sp structure
    n = 0
    for i in range(len(human_virus_species_list)):
        n += len(sp[human_virus_species_list[i]])
    assert(n == np.sum(y))
    # assert(len(sp['non-human']) == (len(y) - n))
    # assert(np.sum(y[sp['non-human']]) == 0)
    # for i in range(len(human_virus_species_list)):
    #     assert(np.all(y[sp[i]]==1)==True)

    model_initializer = make_CNN
    #species_aware_CV(model_initializer, X, y, species, human_virus_species_list, epochs=10, output_string="test00", remove_duplicate_species=True)
    thresh, data = LOOCV(model_initializer, X, y, species, epochs=5, output_string="test", desired_precision=6/7.0)

def test_kuzmin2():
    """
    Compare legacy load with load that performs encoding from the Sequence class.
    """
    # Legacy
    start_time = datetime.datetime.now()
    X, y, species, deflines, sequences, sp, human_virus_species_list, seqs = load_kuzmin_data(representation_type='legacy')
    time_end = datetime.datetime.now()
    time_seconds = (time_end - start_time).total_seconds()
    print("Legacy load took %.1f seconds" % (time_seconds,))

    # New load
    start_time = datetime.datetime.now()
    X2, y2, species2, deflines2, sequences2, sp2, human_virus_species_list2, seqs2 = load_kuzmin_data()
    time_end = datetime.datetime.now()
    time_seconds = (time_end - start_time).total_seconds()
    print("New load took %.1f seconds" % (time_seconds,))

    assert(np.isclose(np.mean(X), np.mean(X2)))
    assert (np.isclose(np.mean(y), np.mean(y2)))

if __name__ == "__main__":
    set_data_directory("test_kuzmin_LOOCV_updates3")
    test_kuzmin()
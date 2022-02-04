"""
Test and compare labeling options in load_kuzmin_data()
"""
from mgm.common.utils import set_data_directory
from mgm.data.kuzmin_data import load_kuzmin_data, species_aware_CV, LOOCV
from mgm.models.NN import make_LSTM, make_CNN
import numpy as np



def test_kuzmin_labeling():
    # Run data load with original labeling scheme
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
    model_initializer = make_CNN
    LOOCV(model_initializer, X, y, species, epochs=5, output_string="test")

    # Run data load with original labeling scheme
    X2, y2, species2, deflines2, sequences2, sp2, human_virus_species_list2 = load_kuzmin_data(label_type="by_host")
    model_initializer = make_CNN
    LOOCV(model_initializer, X2, y2, species2, epochs=5, output_string="test2")

    # Compare labels
    hosts = []
    for seq in sequences:
        if seq.host_species == 'Human':
            hosts.append(1)
        else:
            hosts.append(0)
    hosts = np.array(hosts)
    assert(len(hosts) == len(y2))
    assert(np.sum(y2) == np.sum(hosts))

    print("passed!")

if __name__ == "__main__":
    set_data_directory("test_kuzmin_labeling")
    test_kuzmin_labeling()
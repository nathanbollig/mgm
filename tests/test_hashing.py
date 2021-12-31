from mgm.algorithms.hotflip import one_hotflip
from mgm.algorithms.mgm import mgm_d
from mgm.algorithms.mutations import variant_search
from mgm.algorithms.utils import compute_gradient
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import load_kuzmin_data
from mgm.models.NN import make_LSTM, make_CNN
import numpy as np

# Load Data
X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
n_positions = X.shape[1]
n_characters = X.shape[2]
assert (n_positions == 2396)

# Find first negative instance in the dataset
for i in range(1, X.shape[0]):
    if y[i] == 0:
        break
    else:
        i += 1

# Create a Sequence object
x = X[i]
seq = Sequence(x, y[i], n_characters=n_characters, n_positions=n_positions)

# Test hashing
hashes = set()
hashes.add(seq.get_hash())
# One sub
char0 = seq.integer_encoded[1]
char1 = (char0+1) % n_characters
new_hash = seq.get_hash_of_sub(1, char1)
seq.sub(1,char1)
assert(seq.get_hash() == new_hash)
hashes.add(new_hash)
# Second sub
char2 = (char1+1) % n_characters
new_hash = seq.get_hash_of_sub(1, char2)
seq.sub(1,char2)
assert(seq.get_hash() == new_hash)
hashes.add(new_hash)
# Revert to first sub
char3 = (char2+1) % n_characters
assert(seq.get_hash_of_sub(1, char0) in hashes)
assert(seq.get_hash_of_sub(1, char1) in hashes)
assert(seq.get_hash_of_sub(1, char2) in hashes)
assert(seq.get_hash_of_sub(1, char3) not in hashes)

print("Passed!")

from mgm.algorithms.hotflip import one_hotflip
from mgm.algorithms.mgm import mgm_d
from mgm.algorithms.mutations import variant_search
from mgm.algorithms.utils import compute_gradient
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import load_kuzmin_data
from mgm.models.NN import make_LSTM
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

# Train Model
X_train = np.delete(X, i, axis=0)
y_train = np.delete(y, i)
model = make_LSTM(X_train, y_train, N_POS=n_positions)
model.fit(X_train, y_train, epochs=2)

# Create a Sequence object
x = X[i]
seq = Sequence(x, y[i], n_characters=n_characters, n_positions=n_positions)

# Run mgm-d and hotflip for one sub
seq_mgm_d, data_mgm_d = mgm_d(seq.copy(), seq, model=model, representation='one-hot', cost_function='squared_difference', lambda_param=1)
seq_hf, data_hf = one_hotflip(seq.copy(), model=model)

# Checks
assert(data_mgm_d['pos_to_change'] == data_hf['pos_to_change'])
assert(data_mgm_d['current_char_idx'] == data_hf['current_char_idx'])
assert(data_mgm_d['new_char_idx'] == data_hf['new_char_idx'])

assert(np.all(seq_mgm_d.integer_encoded == seq_hf.integer_encoded))

# for debugging
#seq_hf2, data_hf2 = one_hotflip(seq_hf, model=model)
#seq_mgm_d2, data_mgm_d2 = mgm_d(seq_mgm_d, seq, model=model, representation='one-hot', cost_function='squared_difference', lambda_param=1e4)

# Compare MGM-D with one-hot and HF-inspired algorithm
hx_hf = variant_search(seq, model=model, confidence_threshold = 0.9, type="hotflip", verbose=True, fixed_iterations=3)
hx_mgm_d = variant_search(seq, model=model, confidence_threshold = 0.9, type="mgm-d", verbose=True, fixed_iterations=3)

assert(hx_hf.is_same_trajectory(hx_mgm_d) is True)

print("Passed!")

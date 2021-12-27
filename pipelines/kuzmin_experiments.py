

"""
Pipelines for Kuzmin data
"""
from mgm.algorithms.mutations import variant_search, variants
from mgm.analysis.risk_assessment import risk_of_variant
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import load_kuzmin_data
from mgm.common.utils import set_data_directory
from mgm.models.NN import make_LSTM
import pickle
import numpy as np
import pandas as pd

def exp1():
    """
    Run HotFlip-inspired MGM on a sequence.
    """
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
    variant = variant_search(seq, model=model, confidence_threshold = 0.9, type="hotflip", verbose=True)

    # Get risk score
    risk_score = risk_of_variant(variant)

    assert(1==0)

def exp2():
    """
    Run MGM-d on a sequence in one-hot encoding.
    """
    # Load Data
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
    n_positions = X.shape[1]
    n_characters = X.shape[2]
    assert (n_positions == 2396)

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
    variant = variant_search(seq, model=model, confidence_threshold=0.9, type="mgm-d", verbose=True)

    # Get risk score
    risk_score = risk_of_variant(variant)

def exp3():
    """
    Take (human and bat-infecting) SARS CoV2 out of training dataset.
    Withhold some negative instances as well.
    Train on remaining sequences and run MGM-d on held-aside pos and neg sequences.
    Compute risk score for all variants and rank them by risk.
    Hypothesis: bat SARS CoV2 will be ranked highly, pos sequences will be ranked highly (like positive controls)
    """

    NUM_NEGATIVES_WITHHELD = 60
    model_initializer = make_LSTM

    # Load Data
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
    n = X.shape[0]
    n_positions = X.shape[1]
    n_characters = X.shape[2]

    # Withhold sequences
    negs_indices = np.random.choice(np.where(y == 0)[0], NUM_NEGATIVES_WITHHELD, replace=False)
    sars_indices = np.nonzero(species == 'SARS_CoV_2')[0]
    withheld_indices = np.concatenate((negs_indices, sars_indices))

    X_withheld = X[withheld_indices]
    y_withheld = y[withheld_indices]
    species_withheld = species[withheld_indices]
    deflines_withheld = [deflines[i] for i in withheld_indices]
    sequences_withheld = [sequences[i] for i in withheld_indices]

    X_train = np.delete(X, withheld_indices, axis=0)
    y_train = np.delete(y, withheld_indices)
    species_train = np.delete(species, withheld_indices)
    deflines_train = [deflines[i] for i in range(len(y)) if i not in withheld_indices]
    sequences_train = [sequences[i] for i in range(len(y)) if i not in withheld_indices]

    # Evaluation of model approach
    species_aware_CV(model_initializer, X_train, y_train, species_train, sp, human_virus_species_list, epochs=10, output_string="7fold_CV")

    # Train a model on full training set
    model = model_initializer(X_train, y_train, n_positions)
    model.fit(X_train, y_train, epochs=10)

    # Run MGM-d
    variants = []
    num_withheld = len(withheld_indices)
    for i in range(num_withheld):
        label = y_withheld[i]
        print("Sequence %i/%i (label: %s):" % (i + 1, num_withheld, label), end="")
        seq = Sequence(X_withheld[i], label, n_positions=n_positions, n_characters=n_characters)
        variant = variant_search(seq, model=model, confidence_threshold=0.9, type="mgm-d", verbose=True)
        variants.append(variant)

    # Rank by risk score
    rows = []
    for i, variant in enumerate(variants):
        row = (variant.variant_risk, variant.variant_risk_type, variant.variant_cost, variant.variant_cost_type,
               species_withheld[i], y_withheld[i], variant.init_pred, variant.substitution_data[-1]['conf'],
               deflines_withheld[i])
        rows.append(row)

    cols = ['Risk score', 'Risk score type', 'Cost', 'Cost type', 'Species', 'Initial label', 'Initial pred',
            'Final Pred', 'defline']

    output_df = pd.DataFrame(rows, columns=cols)
    output_df = output_df.sort_values(by=['Risk score'], ascending=False)
    output_df.to_csv("rankings.csv")

if __name__ == "__main__":
    set_data_directory("exp3_test2")
    exp3()

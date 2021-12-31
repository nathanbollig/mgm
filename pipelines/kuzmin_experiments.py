

"""
Pipelines for Kuzmin data
"""
from mgm.algorithms.mutations import variant_search, variants
from mgm.analysis.risk_assessment import risk_of_variant
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import load_kuzmin_data, species_aware_CV
from mgm.common.utils import set_data_directory
from mgm.models.NN import make_LSTM, make_CNN
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

def exp3(species_to_withhold = 'SARS_CoV_2', validate_model=False, model_initializer = make_LSTM):
    """
    Take (human and bat-infecting) SARS CoV2 out of training dataset.
    Withhold some negative instances as well.
    Train on remaining sequences and run MGM-d on held-aside pos and neg sequences.
    Compute risk score for all variants and rank them by risk.
    Hypothesis: bat SARS CoV2 will be ranked highly, pos sequences will be ranked highly (like positive controls)
    """

    if isinstance(species_to_withhold, str):
        species_to_withhold = [species_to_withhold]

    NUM_NEGATIVES_WITHHELD = 60

    # Load Data
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
    n = X.shape[0]
    n_positions = X.shape[1]
    n_characters = X.shape[2]

    # Withhold sequences
    negs_indices = np.random.choice(np.where(y == 0)[0], NUM_NEGATIVES_WITHHELD, replace=False)
    held_indices = np.nonzero(np.isin(species, species_to_withhold))[0]
    withheld_indices = np.concatenate((negs_indices, held_indices))

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
    if validate_model:
        species_aware_CV(model_initializer, X_train, y_train, species_train, human_virus_species_list, epochs=10, output_string="7fold_CV")

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
    output_df.to_csv("rankings.csv", index=False)

def exp4():
    """
    Train on entire dataset.
    Run MGM-d on training set and store variants.
    """

    model_initializer = make_LSTM

    # Load Data
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
    n = X.shape[0]
    n_positions = X.shape[1]
    n_characters = X.shape[2]

    # Train a model on full training set
    model = model_initializer(X, y, n_positions)
    model.fit(X, y, epochs=4)

    # Run MGM-d
    variants = []
    for i in range(n):
        label = y[i]
        print("Sequence %i/%i (label: %s):" % (i + 1, n, label), end="")
        seq = Sequence(X[i], label, n_positions=n_positions, n_characters=n_characters)
        variant = variant_search(seq, model=model, confidence_threshold=0.99, type="mgm-d", verbose=True)
        variants.append(variant)

        # Store variant
        with open("variant_%i.pkl" % (i,), 'wb') as pfile:
            pickle.dump(variant, pfile, protocol=pickle.HIGHEST_PROTOCOL)

def exp5_MERS():
    """
    Take (human and other animal-infecting) MERS out of training dataset.
    Withhold some negative instances as well.
    Train on remaining sequences and run MGM-d on held-aside pos and neg sequences.
    Compute risk score for all variants and rank them by risk.
    Hypothesis: camel MERS will be ranked highly, pos sequences will be ranked highly (like positive controls)
    """

    exp3(species_to_withhold = 'Middle_East_respiratory_syndrome_coronavirus', validate_model=False)

def exp6_SARS():
    """
    Take (human and other animal-infecting) SARS out of training dataset.
    Withhold some negative instances as well.
    Train on remaining sequences and run MGM-d on held-aside pos and neg sequences.
    Compute risk score for all variants and rank them by risk.
    Hypothesis: camel MERS will be ranked highly, pos sequences will be ranked highly (like positive controls)
    """

    exp3(species_to_withhold = ['Severe_acute_respiratory_syndrome_related_coronavirus', 'SARS_CoV_2'] , validate_model=False)

if __name__ == "__main__":
    set_data_directory("exp3_test_CNN3")
    exp3(validate_model=True, model_initializer=make_CNN)

    # # Run with validation!!
    # # SARS CoV 2
    # set_data_directory("exp3_CoV2_test1")
    # exp3()
    #
    # # MERS
    # set_data_directory("exp5_MERS_test1")
    # exp5_MERS()
    #
    # # SARS
    # set_data_directory("exp6_SARS_test1")
    # exp6_SARS()

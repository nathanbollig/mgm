"""
Spillover simulation utilities

This code is designed and tested for data in the format delivered by load_kuzmin_data.
"""

from mgm.algorithms.mutations import variant_search, variants
from mgm.analysis.risk_assessment import risk_of_variant
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import load_kuzmin_data, species_aware_CV, LOOCV, all_species_index_sets
from mgm.common.utils import set_data_directory
from mgm.models.NN import make_LSTM, make_CNN
import pickle
import numpy as np
import pandas as pd

def spillover_experiment(species_to_withhold = 'SARS_CoV_2', validate_model=False, model_initializer = make_LSTM, desired_precision=0.9):
    # Load data and get variants
    variants = spillover_get_variants(species_to_withhold = species_to_withhold, validate_model=validate_model,
                                      model_initializer = model_initializer, desired_precision=desired_precision)

    # Run analysis of variants (saves files)
    analyze_variants(variants)

def spillover_get_variants(species_to_withhold = 'SARS_CoV_2', validate_model=False, model_initializer = make_LSTM, desired_precision=0.9):
    """
    Withhold a designated species from the training dataset.
    Iterate on all negative species => select confidence threshold using internal LOOCV, train a model, use model to guide mgm on the negative group
    Also one iteration with positive sequences included
    Pool MGM results and return a list of variants
    """

    # Load Data
    X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()
    n = X.shape[0]
    n_positions = X.shape[1]
    n_characters = X.shape[2]

    # Evaluation of model approach
    if validate_model:
        LOOCV(model_initializer, X, y, species, epochs=5, output_string="model_eval")

    # Withhold sequences => X_withheld and X_train
    withheld_indices = np.nonzero(species == species_to_withhold)[0]

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

    # External Validation - negative controls
    variants = []
    sp = all_species_index_sets(species_train)
    for species_name, indices in sp.items():
        if species_name in human_virus_species_list:
            continue  # Only use negative groups
        # Get data for iteration
        X_iteration_val = X_train[indices]
        y_iteration_val = y_train[indices]
        X_iteration_train = np.delete(X_train, indices, axis=0)
        y_iteration_train = np.delete(y_train, indices)
        species_iteration_train = np.delete(species_train, indices)
        # Run iteration
        new_variants = external_CV_iteration(model_initializer, X_iteration_train, y_iteration_train, species_iteration_train,
                                     X_iteration_val, y_iteration_val, desired_precision=desired_precision)
        # Store list of variants
        variants.extend(new_variants)

    # External Validation - validation fold with simulated precursor sequences
    new_variants = external_CV_iteration(model_initializer, X_train, y_train, species_train,
                                         X_withheld, y_withheld, desired_precision=desired_precision)
    variants.extend(new_variants)

    return variants

def external_CV_iteration(model_initializer, X_train, y_train, species_train, X_val, y_val, desired_precision):
    """
    Given a training and validation set, use the training set to select a confidence threshold and train a model.
    Use the trained model to guide mgm on the validation set.
    """

    # Select confidence threshold on training set
    threshold = LOOCV(model_initializer, X_train, y_train, species_train, epochs=5, output_string=None, desired_precision=desired_precision)

    # Train a model on full training set
    n_positions = X_train.shape[1]
    n_characters = X_train.shape[2]
    model = model_initializer(X_train, y_train, n_positions)
    model.fit(X_train, y_train, epochs=10)

    # Run MGM-d
    variants = []
    num_withheld = len(y_val)
    for i in range(num_withheld):
        label = y_val[i]
        print("Sequence %i/%i (label: %s):" % (i + 1, num_withheld, label), end="")
        seq = Sequence(X_val[i], label, n_positions=n_positions, n_characters=n_characters)
        variant = variant_search(seq, model=model, confidence_threshold=threshold, type="mgm-d", verbose=True)
        variants.append(variant)

    return variants

def analyze_variants(variants):
    # Rank by risk score
    rows = []
    for i, variant in enumerate(variants):
        if len(variant.substitution_data) > 0:
            final_pred = variant.substitution_data[-1]['conf']
        else:
            final_pred = variant.init_pred
        row = (variant.variant_risk, variant.variant_risk_type, variant.variant_cost, variant.variant_cost_type,
               species_withheld[i], y_withheld[i], variant.init_pred, final_pred, deflines_withheld[i])
        rows.append(row)

    cols = ['Risk score', 'Risk score type', 'Cost', 'Cost type', 'Species', 'Initial label', 'Initial pred',
            'Final Pred', 'defline']

    output_df = pd.DataFrame(rows, columns=cols)
    output_df = output_df.sort_values(by=['Risk score'], ascending=False)
    output_df.to_csv("rankings.csv", index=False)
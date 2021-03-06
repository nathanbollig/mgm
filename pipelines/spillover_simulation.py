"""
Spillover simulation utilities

This code is designed and tested for data in the format delivered by load_kuzmin_data.
"""

from mgm.algorithms.mutations import variant_search, variants
from mgm.analysis.risk_assessment import risk_of_variant
from mgm.common.cost_functions import num_differences, similarity
from mgm.common.sequence import Sequence
from mgm.data.kuzmin_data import load_kuzmin_data, species_aware_CV, LOOCV, all_species_index_sets
from mgm.common.utils import set_data_directory
from mgm.models.NN import make_LSTM, make_CNN
import pickle
import numpy as np
import pandas as pd

def spillover_experiment(species_to_withhold = 'SARS_CoV_2', representation='one-hot', validate_model=False, model_initializer = make_LSTM, desired_precision=0.9, confidence_threshold=None, fixed_iterations=250):
    # Load data and get variants
    variants, seqs = spillover_get_variants(species_to_withhold = species_to_withhold, representation=representation, validate_model=validate_model,
                                      model_initializer = model_initializer, desired_precision=desired_precision, confidence_threshold=confidence_threshold,
                                      fixed_iterations=fixed_iterations)

    # Save variants
    with open("variants.pkl", 'wb') as file:
        pickle.dump(variants, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Extract non-withheld positive sequences for baseline computations
    non_withheld_pos_seqs = select_non_withheld_pos(seqs, species_withheld= species_to_withhold)

    # Save non-withheld pos seqs
    with open("non_withheld_pos_seqs.pkl", 'wb') as file:
        pickle.dump(non_withheld_pos_seqs, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Run analysis of variants (saves files)
    analyze_variants(variants, non_withheld_pos_seqs)

def spillover_get_variants(representation, fixed_iterations, species_to_withhold = 'SARS_CoV_2', validate_model=False, model_initializer = make_LSTM, desired_precision=6/7.0, confidence_threshold=None):
    """
    Withhold a designated species from the training dataset.
    Iterate on all negative species => select confidence threshold using internal LOOCV, train a model, use model to guide mgm on the negative group
    Also one iteration with positive sequences included
    Pool MGM results and return a list of variants + all Sequence objects returned from data load.
    """

    # Load Data
    X, y, species, deflines, sequences, sp, human_virus_species_list, seqs = load_kuzmin_data(representation_type=representation)
    num_all_species = len(sp.keys())
    n = X.shape[0]
    n_positions = X.shape[1]
    n_characters = X.shape[2]

    # Evaluation of model approach
    if validate_model:
        _, model_eval_data = LOOCV(model_initializer, X, y, species, epochs=5, output_string="model_eval")

    # Withhold sequences => X_withheld and X_train
    withheld_indices = np.nonzero(species == species_to_withhold)[0]
    assert(len(withheld_indices) == len(sp[species_to_withhold]))

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
    i = 1
    sp_train = all_species_index_sets(species_train)
    assert(len(sp_train.items()) == num_all_species - 1)
    iteration_train_size = 0
    iteration_val_size = 0
    for species_name, indices in sp_train.items():
        if species_name in human_virus_species_list:
            continue  # Only use negative groups
        # Get data for iteration
        X_iteration_val = X_train[indices]
        y_iteration_val = y_train[indices]
        X_iteration_train = np.delete(X_train, indices, axis=0)
        y_iteration_train = np.delete(y_train, indices)
        species_iteration_train = np.delete(species_train, indices)
        deflines_iteration_train = [deflines_train[i] for i in range(len(y_train)) if i not in indices]
        species_iteration_val = species_train[indices]
        deflines_iteration_val = [deflines_train[i] for i in indices]

        # Tests for correctness
        assert(len(all_species_index_sets(species_iteration_train).keys()) == num_all_species - 2)  # the number of species on the iteration's training set should be total species minus 2 (1 group withheld for experiment, 1 group held for this fold of external CV)
        assert(X_iteration_train.shape[0] == len(y_iteration_train))
        assert(len(y_iteration_train) == len(species_iteration_train))
        assert(X_iteration_val.shape[0] == len(y_iteration_val))
        iteration_train_size += len(y_iteration_train)
        iteration_val_size += len(y_iteration_val)

        # Run iteration
        new_variants = external_CV_iteration(model_initializer, X_iteration_train, y_iteration_train, species_iteration_train, X_iteration_val, y_iteration_val,
                                            species_iteration_val, deflines_iteration_val, desired_precision=desired_precision,
                                            confidence_threshold=confidence_threshold, fixed_iterations=fixed_iterations)
        i += 1
        # Store list of variants
        variants.extend(new_variants)

    # Tests for correctness
    size_non_human_groups = 0
    size_human_not_withheld = 0
    for key in sp:
        if key not in human_virus_species_list:
            size_non_human_groups += len(sp[key])
        elif key != species_to_withhold:
            size_human_not_withheld += len(sp[key])
    assert(size_non_human_groups == iteration_val_size)
    assert((num_all_species - 7 - 1) * size_non_human_groups + (num_all_species - 7) * size_human_not_withheld == iteration_train_size)
    assert(len(variants) == size_non_human_groups)

    # External Validation - validation fold with simulated precursor sequences
    new_variants = external_CV_iteration(model_initializer, X_train, y_train, species_train, X_withheld, y_withheld,
                                         species_withheld, deflines_withheld, desired_precision=desired_precision,
                                         confidence_threshold=confidence_threshold, fixed_iterations=fixed_iterations)
    variants.extend(new_variants)

    # Tests for correctness
    assert(len(variants) == size_non_human_groups + len(y_withheld))

    return variants, seqs

def external_CV_iteration(model_initializer, X_train, y_train, species_train, X_val, y_val, species_val, deflines_val, desired_precision, confidence_threshold, fixed_iterations):
    """
    Given a training and validation set, use the training set to select a confidence threshold and train a model.
    Use the trained model to guide mgm on the validation set.
    """

    # Select confidence threshold on training set
    # TODO: set threshold_only to true when done with testing
    if confidence_threshold is not None:
        threshold = confidence_threshold
        data = None
    else:
        threshold, data = LOOCV(model_initializer, X_train, y_train, species_train, epochs=5, desired_precision=desired_precision, threshold_only=False)


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
        # Create sequence object
        seq = Sequence(X_val[i], label, n_positions=n_positions, n_characters=n_characters)
        seq.set_species(species_val[i])
        seq.set_defline(deflines_val[i])
        # Run variant search
        variant = variant_search(seq, model=model, confidence_threshold=threshold, type="mgm-d", verbose=True, fixed_iterations=fixed_iterations)
        if data is not None:
            variant.set_fields(LOOCV_data = data)
        variants.append(variant)

    return variants

def diff_from_nearest(seq, reference_seqs):
    """
    Given an input Sequence, find the distance to the closest member of a list of reference Sequences. Difference
    is edit distance (number of amino acid differences) and Blossom similarity.
    """
    edit_dist_nearest, sim_nearest = None, None
    for ref in reference_seqs:
        diff = num_differences(seq, ref)
        sim = similarity(seq, ref)
        if edit_dist_nearest is None or diff < edit_dist_nearest:
            edit_dist_nearest = diff
        if sim_nearest is None or sim < sim_nearest:
            sim_nearest = sim
    return edit_dist_nearest, sim_nearest

def select_non_withheld_pos(sequences, species_withheld = 'SARS_CoV_2'):
    non_withheld_pos_seqs = []
    for seq in sequences:
        if seq.y == 1 and seq.get_species() != species_withheld:
            non_withheld_pos_seqs.append(seq)
    return non_withheld_pos_seqs

def analyze_variants(variants, non_withheld_pos_seqs, filename="rankings_%s_keepfinal.csv"):
    # # Recapitulate withheld group
    # withheld_seqs = []
    # for variant in variants:
    #     if variant.init_seq.y == 1:
    #         withheld_seqs.append(variant.init_seq)

    # Rank by risk score
    rows = []
    for i, variant in enumerate(variants):
        if len(variant.substitution_data) > 0:
            final_pred = variant.get_final_pred()
        else:
            final_pred = variant.init_pred
        edit_dist_nearest, sim_nearest = diff_from_nearest(variant.init_seq, non_withheld_pos_seqs)
        row = (variant.variant_risk, variant.variant_risk_type, variant.variant_cost, variant.variant_cost_type, num_differences(variant.init_seq, variant.final_seq), edit_dist_nearest, sim_nearest,
               variant.init_seq.get_species(), variant.init_seq.y, variant.init_pred, final_pred, variant.confidence_threshold, variant.init_seq.get_defline())
        rows.append(row)

    cols = ['Risk score', 'Risk score type', 'Cost', 'Cost type', 'Num Differences', 'Edit Distance to Closest Positive', 'Blossom Similarity to Closest Positive', 'Species', 'Initial label', 'Initial pred',
            'Final Pred', 'Threshold', 'defline']

    output_df = pd.DataFrame(rows, columns=cols)
    output_df['Risk score'] = pd.to_numeric(output_df['Risk score'], errors='coerce')
    output_df = output_df.sort_values(by=['Risk score'], ascending=False)
    output_df['Risk score'] = output_df['Risk score'].fillna("undefined")

    if filename == "rankings_%s_keepfinal.csv":
        filename = filename % (str(int(variants[0].confidence_threshold * 100)),)
    output_df.to_csv(filename, index=False)

def truncate_mutation_trajectory(substitution_data, confidence_threshold):
    for i, sub_dict in enumerate(substitution_data):
        if sub_dict['pred_proba'] > confidence_threshold:
            substitution_data_truncated = substitution_data[:i+1]
            return substitution_data_truncated
    return substitution_data

def analyze_variants_at_threshold(variants, non_withheld_pos_seqs, THRESHOLD, rankings_path, keep_final_seq=False):

    for variant in variants:
        if THRESHOLD < variant.confidence_threshold:
            variant.confidence_threshold = THRESHOLD
            variant.substitution_data = truncate_mutation_trajectory(variant.substitution_data, variant.confidence_threshold)
            if keep_final_seq == False:
                variant.final_seq = variant.replay_trajectory()
            variant.compute_cost()

    analyze_variants(variants, non_withheld_pos_seqs, filename=rankings_path)

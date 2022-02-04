# -*- coding: utf-8 -*-
"""
Create Kuzmin dataset

This file is adapted from https://github.com/kuzminkg/CoVs-S-pr

Kuzmin K, Adeniyi AE, DaSouza AK, Lim D, Nguyen H, Molina NR, et al. Machine learning methods
accurately predict host specificity of coronaviruses based on spike sequences alone.
Biochem Biophys Res Commun. 2020;533: 553–558. doi:10.1016/j.bbrc.2020.09.010
"""

import numpy as np
from Bio import SeqIO
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, roc_curve, \
    roc_auc_score, precision_recall_curve
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.utils import shuffle
import pandas as pd
from mgm.common.utils import get_full_path
input_file_name = get_full_path("data", "kuzmin.fasta")

human_virus_species_set = {'Human_coronavirus_NL63', 'Betacoronavirus_1',
                                'Human_coronavirus_HKU1', 'Severe_acute_respiratory_syndrome_related_coronavirus',
                                'SARS_CoV_2', 'Human_coronavirus_229E', 'Middle_East_respiratory_syndrome_coronavirus'}
human_virus_species_list = list(human_virus_species_set)
human_virus_species_list.sort()

alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ-'
aa_vocab = list(alphabet)[:-1]

class fasta_sequence:
    def __init__(self, defline, sequence, target, type_of_encoding="onehot"):
        self.defline = defline
        self.sequence = sequence
        self.target = target

        # Parse info from defline
        self.strain_name = defline.split("|")[0]
        self.accession_number = defline.split("|")[1]
        self.host_species = defline.split("|")[2]
        self.virus_species = defline.split("|")[3]

        """
        We convert a string with the alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ-'
        into either a list mapping chars to integers (called integer encoding),
        or a one-hot encoding. In the latter, each amino acid is represented as an one-hot vector of length 25,
        where each position, except one, is set to 0.  E.g., alanine is encoded as 10000000000000000000,
        cystine is encoded as 01000000000000000000.
        Symbol '-' is encoded as a zero-vector.
        """

        def encoding(sequence, type_of_encoding):
            # define a mapping of chars to integers
            char_to_int = dict((c, i) for i, c in enumerate(alphabet))

            # integer encoding
            integer_encoded = [char_to_int[char] for char in sequence]

            # one-hot encoding
            onehot_encoded = list()
            for value in integer_encoded:
                letter = [0 for _ in range(len(alphabet) - 1)]
                if value != len(alphabet) - 1:
                    letter[value] = 1
                onehot_encoded.append(letter)
            flat_list = [item for sublist in onehot_encoded for item in sublist]

            if type_of_encoding == "onehot":
                return flat_list
            else:
                return integer_encoded

        #  last step for constructor is to compute and save the encoding in the fasta_sequence object
        self.encoded = encoding(sequence, type_of_encoding)

def read_fasta(input_file_name):
    """
    Read the fasta-file. Returns the following as parallel lists:
       1. deflines - description of a sequence, in format: Strain Name | Accession Number | Host Species | Virus Species
       2. protein_sequences - biopython Seq object
       3. targets - integer label (0 or 1)
    """
    # Form dictionary from fasta file
    sequences_dictionary = {sequence.id : sequence.seq for sequence in SeqIO.parse(input_file_name,'fasta')}

    # Deflines
    deflines = [entry for entry in sequences_dictionary.keys()]

    # Protein sequences
    protein_sequences = [entry for entry in sequences_dictionary.values()]

    # Targets: we assign 1 iff a virus species is one from the 7 human-infective
    targets = [0]*len(deflines)
    for i, defline in enumerate(deflines):
        for virus_species in human_virus_species_set:
            if virus_species in defline:
                targets[i] = 1

    return deflines, protein_sequences, targets


def get_data(list_of_sequences, label_type):
    """
    Iterate through a list of fasta_sequence objects and retrieve lists of encoded sequences, targets, and species

    The definition of binary target is determined by the label_type.
    """
    list_of_encoded_sequences = []
    list_of_targets = []
    list_of_species = []
    for entry in list_of_sequences:
        list_of_encoded_sequences.append(entry.encoded)
        if label_type == "by_species":
            list_of_targets.append(entry.target)
        if label_type == "by_host":
            label = int(entry.host_species == 'Human')
            list_of_targets.append(label)
        list_of_species.append(entry.virus_species)

    return list_of_encoded_sequences, list_of_targets, list_of_species


def index_sets(human_virus_species_list, species):
    """
    Retrieves indices of human-infecting species and all non-human sequences in one group - suitable for 7-fold CV.

    The data loading approach uses a set of parallel lists (deflines, protein_sequences, etc.). With respect
    to the common ordering of data items, we want to identify the set of indices for each human-infecting species.

    Inputs:
        human_virus_species_list - a list of human-infecting species
        species - list of species labels for each data item (e.g. parallel to deflines)

    Output:
        sp - dictionary such that sp[index of species in human_virus_species_list] is the set of indices corresponding to
                that species in the common ordering of data items. In addition, the value of sp['non-human'] is a
                list of indices for non-human-infecting viruses.
    """
    sp = {}
    for i in range(len(human_virus_species_list)):
        sp[i] = []
    sp['non-human'] = []

    # Populate the dictionary
    for i in range(len(species)):
        if species[i] in human_virus_species_list:
            species_idx = human_virus_species_list.index(species[i])
            sp[species_idx].append(i)
        else:
            sp['non-human'].append(i)

    return sp

def load_kuzmin_data(label_type="by_species"):
    """
    Main function for loading the dataset of Kuzmin et al.

    Input:
        label_type - "by_species" (+ if virus species infects human) or "by_host" (+ if isolated from human host)

    Returns the following objects. The first four are parallel lists:
        X - one-hot encoded sequences, dataset shape (1238, 2396, 25)
        y - 1d int array (of 0s and 1s) of length 1238
        species - 1d string array of species labels of length 1238
        deflines - list of deflines of length 1238
        sequences - list of fasta_sequence objects of length 1238
        sp - index set dictionary as returned by all_species_index_sets()
        human_virus_species_list - list of human infecting species labels (indices are keys in `sp`)
    """
    # Read fasta file
    deflines, protein_sequences, targets = read_fasta(input_file_name)

    # Create a list of fasta_sequence objects
    sequences = []
    for i in range(len(deflines)):
        seq = fasta_sequence(deflines[i], protein_sequences[i], targets[i])
        sequences.append(seq)

    # Get data from sequence objects
    X, y, species = get_data(sequences, label_type=label_type)

    # Convert data to numpy arrays and set shape
    N_POS = 2396
    N_CHAR = 25
    X = np.array(X).reshape((-1, N_POS, N_CHAR))
    y = np.array(y)
    species = np.array(species)

    # Randomize ordering
    X, y, species, deflines, sequences = shuffle(X, y, species, deflines, sequences)

    # Get index sets
    sp = all_species_index_sets(species)

    return X, y, species, deflines, sequences, sp, human_virus_species_list

########################################################################################################################
# Evaluation code written for models using this data
########################################################################################################################
def all_species_index_sets(species):
    """
    Builds index of indices for each unique species in the species list.

    Inputs:
        species - list of species labels for each data item (e.g. parallel to deflines)

    Output:
        species_index - dictionary such that sp[species name] is a list of indices corresponding to
                that species in the species list.
    """

    species_index = {}

    for i, species_name in enumerate(species):
        if species_name not in species_index.keys():
            species_index[species_name] = []
        species_index[species_name].append(i)

    return species_index

def LOOCV(model_initializer, X, y, species, epochs=1, output_string="test", desired_precision=None):
    """
    Takes in a model initializer and kuzmin data, applies leave-one-out CV (wrt viral species) to determine model performance and threshold for desired precision.

    Strategy:
        1. Iterate over unique viral species (54+7 in total for whole dataset)
        2. For each species, withhold all sequences of that species.
        3. Train a model on the remaining sequences
        4. Apply the trained model to each of the withheld sequences, to get a set of predicted probabilities.
        5. Report average predicted probability for each class within that viral species (average in csv file + distributions shown in violin plot)
        6. Use this set of numbers with corresponding binary targets to generate an ROC curve.

    Inputs:
        Model initializer - a function that takes three parameters: X, y, N_POS
        X, y, species - data, labels, and a parallel list of species
        desired_precision - determines deserved level of model precision for the computed prediction threshold
        output_string - file prefix; if None, no files are saved

    Output:
        Saves data files and images for model evaluation
        Returns smallest prediction threshold at which at least the desired precision is reached

    """
    # Build dictionary of index sets for each species
    species_index = all_species_index_sets(species)

    # Get list of all unique species names
    species_names_list = np.array(list(species_index.keys()))

    # Do LOOCV on species_names_list
    loo = LeaveOneOut()
    num_splits = loo.get_n_splits(species_names_list)
    print("Running LOOCV with %i splits..." % (num_splits,))

    output = []
    i = 0
    Y_avg_proba = []
    Y_pred_lists = []
    Y_targets = []
    Y_species = []

    for train_species_idx, hold_species_idx in loo.split(species_names_list):
        train_species = species_names_list[train_species_idx] # names of species to be in training set
        hold_species = species_names_list[hold_species_idx] # name of withheld species

        assert (len(hold_species) == 1)
        hold_species = hold_species[0]

        # Build train index set (into X and y)
        train = []
        for name in train_species:
            index_set = species_index[name] # list of indices where this species is
            train.extend(index_set)

        # Built test index set (into X and y)
        test = species_index[hold_species]

        # Create train and test set
        X_train = X[train]
        y_train = y[train]
        species_train = species[train]
        X_test = X[test]
        y_test = y[test]
        species_test = species[test]

        print("*******************FOLD %i: %s*******************" % (i + 1, hold_species))
        test_size = len(y_test)
        print("Test size = %i" % (test_size,))

        # Check if one label on test fold, e.g. if labels are by species
        one_test_label = np.all(y_test == y_test[0])

        if one_test_label:
            test_label = y_test[0]
            print("Test label = %i" % (test_label,))

        # Train model
        model = model_initializer(X_train, y_train, N_POS=X.shape[1])
        model.fit(X_train, y_train, epochs=epochs)

        # Apply model to test set
        y_proba = model.predict(X_test)
        pred_list = y_proba.flatten()
        assert(len(pred_list) == len(y_test))

        if one_test_label:
            mean_pred = np.mean(pred_list)
            # Store fold results
            output.append((i, hold_species, test_label, test_size, mean_pred, pred_list))
            Y_avg_proba.append(mean_pred)
            Y_targets.append(test_label)
            Y_species.append(hold_species)
            Y_pred_lists.append(pred_list)
        else:
            pred_list0 = pred_list[y_test == 0]
            pred_list1 = pred_list[y_test == 1]
            # Store 0 target results on fold
            output.append((i, hold_species, 0, len(pred_list0), np.mean(pred_list0), pred_list0))
            Y_avg_proba.append(np.mean(pred_list0))
            Y_targets.append(0)
            Y_species.append(hold_species)
            Y_pred_lists.append(pred_list0)
            # Store 1 target results on fold
            output.append((i, hold_species, 1, len(pred_list1), np.mean(pred_list1), pred_list1))
            Y_avg_proba.append(np.mean(pred_list1))
            Y_targets.append(1)
            Y_species.append(hold_species)
            Y_pred_lists.append(pred_list1)

        i += 1

    # Save fold summary
    output_df = pd.DataFrame(output, columns=['fold', 'species', 'target_label', 'test_size', 'mean_pred', 'pred_list'])
    print(output_df)
    if output_string is not None:
        output_df.to_csv('%s_results.csv' % (output_string,), index=False)

    # Calculate threshold for desired precision
    output_threshold = None
    if desired_precision is not None:
        output_threshold = 1
        precision, recall, thresholds = precision_recall_curve(Y_targets, Y_avg_proba)
        for i in range(len(thresholds)):
            if precision[i] >= desired_precision:
                output_threshold = thresholds[i]
                break

    # Generate ROC curve
    def get_ROC(Y_targets, Y_proba):
        fpr, tpr, _ = roc_curve(Y_targets, Y_proba)
        try:
            auc = roc_auc_score(Y_targets, Y_proba)
        except ValueError:
            auc = 0
        return fpr, tpr, auc

    # Save model eval output
    if output_string is not None:
        fpr, tpr, auc = get_ROC(Y_targets, Y_avg_proba)
        plt.step(fpr, tpr, where='post', label='(AUC=%.2f)' % (auc,))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (based on ranking mean pred on each viral species)')
        plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
        plt.savefig("%s_ROC.jpg" % (output_string,), dpi=400, bbox_inches="tight")
        plt.clf()

        # Violin plot
        fig, ax = plt.subplots()
        parts = plt.violinplot(Y_pred_lists, vert=False, widths=0.9)

        # Set human-infecting violins to red color
        for i, violin in enumerate(parts['bodies']):
            if Y_targets[i] == 1:
                violin.set_facecolor('red')
                #violin.set_edgecolor('red')

        # Set lines to red color
        cbars_color = parts['cbars'].get_color()[0]

        colors = []
        for i in range(len(Y_targets)):
            if Y_targets[i] == 1:
                colors.append(np.array(to_rgba('red')))
            else:
                colors.append(cbars_color)
        parts['cbars'].set_color(colors)
        parts['cmaxes'].set_color(colors)
        parts['cmins'].set_color(colors)

        # Label y axis
        ax.yaxis.set_tick_params(direction='out')
        ax.set_yticks(np.arange(1, len(Y_species) + 1), labels=Y_species)
        ax.set_ylim(0.25, len(Y_species) + 0.75)
        ax.set_ylabel('Virus species')

        # Other plot setup
        ax.set_xlabel('Model prediction')
        ax.set_title('Predictions on each holdout set')
        ax.yaxis.grid(linewidth=1, linestyle='--')

        # Legend
        red_patch = mpatches.Patch(color='red', alpha=0.3, label='Human-infecting')
        blue_patch = mpatches.Patch(color='blue', alpha=0.3, label='Non-human-infecting')
        plt.legend(handles=[red_patch, blue_patch], loc='upper left', fontsize=12, bbox_to_anchor=(1.05, 1))

        # Save plot
        fig.set_size_inches(13, 12)
        fig.savefig("%s_preds.jpg" % (output_string,), dpi=600, bbox_inches="tight")
        plt.clf()

    return output_threshold

def species_aware_CV(model_initializer, X, y, species, human_virus_species_list, epochs=1, output_string="test", remove_duplicate_species=False):
    """
    Takes in a model initializer and kuzmin data, applies species-aware 7-fold CV to determine model performance.

    Model initializer parameter is a function that takes three parameters: X, y, N_POS
    """

    sp = index_sets(human_virus_species_list, species)

    def evaluate(y_proba, y_test, y_proba_train, y_train, verbose=False):
        # PR curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        #    fig, ax = plt.subplots()
        #    ax.plot(recall, precision)
        #    ax.set(xlabel='Recall', ylabel='Precision', title=model_name + ' (AP=%.3f)' % (ap,))
        #    ax.grid()
        #    fig.savefig(model_name+"%i_pr_curve.jpg" % (int(time.time()),), dpi=500)
        #    plt.show()

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0

        #    fig, ax = plt.subplots()
        #    ax.plot(fpr, tpr)
        #    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=model_name + ' (AUC=%.3f)' % (auc,))
        #    ax.grid()
        #    #fig.savefig(model_name + "_roc_curve.jpg", dpi=500)
        #    plt.show()

        # Evaluate on train set
        train_accuracy = accuracy_score(y_train, (y_proba_train >= 0.5).astype(int))
        train_recall = recall_score(y_train, (y_proba_train >= 0.5).astype(int))
        train_precision = precision_score(y_train, (y_proba_train >= 0.5).astype(int))
        train_f1 = f1_score(y_train, (y_proba_train >= 0.5).astype(int))

        # Evaluate on validation set
        test_accuracy = accuracy_score(y_test, (y_proba >= 0.5).astype(int))
        test_recall = recall_score(y_test, (y_proba >= 0.5).astype(int))
        test_precision = precision_score(y_test, (y_proba >= 0.5).astype(int))
        test_f1 = f1_score(y_test, (y_proba >= 0.5).astype(int))

        if verbose == True:
            print('Train Accuracy: %.2f' % (train_accuracy * 100))
            print('Train Recall: %.2f' % (train_recall * 100))
            print('Train Precision: %.2f' % (train_precision * 100))
            print('Train F1: %.2f' % (train_f1 * 100))
            print('Test Accuracy: %.2f' % (test_accuracy * 100))
            print('Test Recall: %.2f' % (test_recall * 100))
            print('Test Precision: %.2f' % (test_precision * 100))
            print('Test F1: %.2f' % (test_f1 * 100))

        return ap, auc, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1

    # Main
    kfold = GroupKFold(n_splits=7)

    Y_targets = []
    Y_species = []
    output = []
    i = 0
    Y_proba = []

    # Collect data for testing
    X_TRAIN = []
    X_TEST = []
    Y_TRAIN = []
    Y_TEST = []

    for train, test in kfold.split(X[sp['non-human']], y[sp['non-human']],
                                   species[sp['non-human']]):  # start by splitting only non-human data
        # Put the ith human-infecting virus species into the test set, the rest into train
        # Get indices of training species
        training_species = [k for k in [0, 1, 2, 3, 4, 5, 6] if k != i]
        training_species_idx = []
        for j in training_species:
            training_species_idx.extend(sp[j])

        # Create train and test arrays by concatenation
        X_train = np.vstack((X[sp['non-human']][train], X[training_species_idx]))
        X_test = np.vstack((X[sp['non-human']][test], X[sp[i]]))
        y_train = np.concatenate((y[sp['non-human']][train], y[training_species_idx]))
        y_test = np.concatenate((y[sp['non-human']][test], y[sp[i]]))
        y_test_species = np.concatenate((np.zeros((len(test),), dtype=int), np.full((len(y[sp[i]]),), i + 1,
                                                                                    dtype=int)))  # 0 for non-human, 1-based index for human
        species_test = np.concatenate((species[sp['non-human']][test], species[sp[i]]))
        num_test_nonhuman_species = np.unique(species[sp['non-human']][test]).size


        # Shuffle arrays
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test, y_test_species = shuffle(X_test, y_test, y_test_species)

        # Filter out duplicate species in test set
        if remove_duplicate_species:
            assert(X_test.shape[0] == len(y_test))
            assert(len(y_test) == len(species_test))
            # If we want to, here we could filter modify y_test and X_test according to desired deduplication logic

        # Store data for testing
        X_TRAIN.append(X_train)
        X_TEST.append(X_test)
        Y_TRAIN.append(y_train)
        Y_TEST.append(y_test)

        print("*******************FOLD %i: %s*******************" % (i + 1, human_virus_species_list[i]))
        print("Test size = %i" % (len(y_test),))
        print("Test non-human size = %i" % (len(X[sp['non-human']][test])), )
        print("Test non-human # species = %i" % (num_test_nonhuman_species,))
        print("Test human size = %i" % (len(X[sp[i]]),))
        print("Test pos class prevalence: %.3f" % (np.mean(y_test),))

        model = model_initializer(X_train, y_train, N_POS=X.shape[1])
        model.fit(X_train, y_train, epochs=epochs)
        y_proba = model.predict(X_test)
        y_proba_train = model.predict(X_train)
        results = evaluate(y_proba, y_test, y_proba_train, y_train)
        output.append((i,) + results)
        Y_proba.extend(y_proba)

        Y_targets.extend(y_test)
        Y_species.extend(y_test_species)
        i += 1

    print("*******************SUMMARY*******************")

    output_df = pd.DataFrame(output,
                             columns=['Fold', 'ap', 'auc', 'train_accuracy', 'train_recall',
                                      'train_precision', 'train_f1', 'test_accuracy', 'test_recall', 'test_precision',
                                      'test_f1'])
    print(output_df)
    output_df.to_csv('%s_results.csv' % (output_string,))

    # Generate pooled ROC curve
    auc_baseline = roc_auc_score(Y_targets, np.ones(len(Y_targets)))

    def get_ROC(Y_targets, Y_proba):
        fpr, tpr, _ = roc_curve(Y_targets, Y_proba)
        try:
            auc = roc_auc_score(Y_targets, Y_proba)
        except ValueError:
            auc = 0
        return fpr, tpr, auc

    fpr, tpr, auc = get_ROC(Y_targets, Y_proba)
    plt.step(fpr, tpr, where='post', label='(AUC=%.2f)' % (auc,))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Sequence classification performance; baseline AUC=%.3f' % (auc_baseline,))
    plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
    plt.savefig("%s_ROC.jpg" % (output_string,), dpi=400, bbox_inches="tight")
    plt.clf()

    # Generate pooled PR curve
    Y_vals = []
    for i in Y_species:
        if i==0:
            Y_vals.append('Non-human')
        else:
            Y_vals.append(human_virus_species_list[i-1])

    plt.scatter(Y_proba, Y_vals, facecolors='none', edgecolors='r')
    plt.xlabel('Predicted probability')
    plt.ylabel('Species')
    plt.title('Predicted Probability vs. Infecting Species')
    plt.savefig("%s_species.jpg" % (output_string,), dpi=400, bbox_inches="tight")
    plt.clf()

    return output_df

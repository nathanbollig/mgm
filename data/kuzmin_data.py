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
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, roc_curve, \
    roc_auc_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
import pandas as pd
from mgm.common.utils import get_full_path
input_file_name = get_full_path("data", "kuzmin.fasta")

human_virus_species_set = {'Human_coronavirus_NL63', 'Betacoronavirus_1',
                                'Human_coronavirus_HKU1', 'Severe_acute_respiratory_syndrome_related_coronavirus',
                                'SARS_CoV_2', 'Human_coronavirus_229E', 'Middle_East_respiratory_syndrome_coronavirus'}
human_virus_species_list = list(human_virus_species_set)
human_virus_species_list.sort()

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

            # define character alphabet
            alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ-'
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


def get_data(list_of_sequences):
    """
    Iterate through a list of fasta_sequence objects and retrieve lists of encoded sequences, targets, and species
    """
    list_of_encoded_sequences = []
    list_of_targets = []
    list_of_species = []
    for entry in list_of_sequences:
        list_of_encoded_sequences.append(entry.encoded)
        list_of_targets.append(entry.target)
        list_of_species.append(entry.virus_species)

    return list_of_encoded_sequences, list_of_targets, list_of_species


def index_sets(human_virus_species_list, species):
    """
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

def load_kuzmin_data():
    """
    Main function for loading the dataset of Kuzmin et al.

    Returns the following objects. The first four are parallel lists:
        X - one-hot encoded sequences, dataset shape (1238, 2396, 25)
        y - 1d int array (of 0s and 1s) of length 1238
        species - 1d string array of species labels of length 1238
        deflines - list of deflines of length 1238
        sequences - list of fasta_sequence objects of length 1238
        sp - index set dictionary as returned by index_sets()
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
    X, y, species = get_data(sequences)

    # Convert data to numpy arrays and set shape
    N_POS = 2396
    N_CHAR = 25
    X = np.array(X).reshape((-1, N_POS, N_CHAR))
    y = np.array(y)
    species = np.array(species)

    # Randomize ordering
    X, y, species, deflines, sequences = shuffle(X, y, species, deflines, sequences)

    # Get index sets
    sp = index_sets(human_virus_species_list, species)

    return X, y, species, deflines, sequences, sp, human_virus_species_list

def species_aware_CV(model_initializer, X, y, species, sp, human_virus_species_list, epochs=1, output_string="test"):
    """
    Takes in a model initializer and kuzmin data, applies species-aware 7-fold CV to determine model performance.

    Model initializer parameter is a function that takes three parameters: X, y, N_POS
    """

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
        # Shuffle arrays
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test, y_test_species = shuffle(X_test, y_test, y_test_species)

        # Store data for testing
        X_TRAIN.append(X_train)
        X_TEST.append(X_test)
        Y_TRAIN.append(y_train)
        Y_TEST.append(y_test)

        print("*******************FOLD %i: %s*******************" % (i + 1, human_virus_species_list[i]))
        print("Test size = %i" % (len(y_test),))
        print("Test non-human size = %i" % (len(X[sp['non-human']][test])), )
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
    plt.scatter(Y_proba, Y_species, facecolors='none', edgecolors='r')
    plt.xlabel('Predicted probability')
    plt.ylabel('Species')
    plt.title('Predicted Probability vs. Infecting Species')
    plt.savefig("%s_species.jpg" % (output_string,), dpi=400, bbox_inches="tight")
    plt.clf()

    return output_df

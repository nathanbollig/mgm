"""
Use load_kuzmin_data to retrieve data objects, and trains a LSTM using the species-aware 7-fold CV split.
Training only for 1 epoch with large batch size to increase training speed.

Takes ~ 7 min
"""

from mgm.data.kuzmin_data import load_kuzmin_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Run data load
X, y, species, deflines, sequences, sp, human_virus_species_list = load_kuzmin_data()

# Test sp structure
n = 0
for i in range(len(human_virus_species_list)):
    n += len(sp[i])
assert(n == np.sum(y))
assert(len(sp['non-human']) == (len(y) - n))
assert(np.sum(y[sp['non-human']]) == 0)
for i in range(len(human_virus_species_list)):
    assert(np.all(y[sp[i]]==1)==True)


# Train a model using species-aware CV splitting

def make_LSTM(X_train, y_train, N_POS):
    n = X_train.shape[0]
    X_train = X_train.reshape((n, N_POS, -1))
    num_features = X_train.shape[2]

    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(N_POS, num_features)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def run_LSTM(X_train, y_train, X_test, N_POS=2396):

    X_train = X_train.reshape((X_train.shape[0], N_POS, -1))
    X_test = X_test.reshape((X_test.shape[0], N_POS, -1))
    model = make_LSTM(X_train, y_train, N_POS=N_POS)
    model.fit(X_train, y_train, epochs=1, batch_size=16)
    y_proba = model.predict(X_test)
    y_proba_train = model.predict(X_train)

    return y_proba, y_proba_train

def evaluate(y_proba, y_test, y_proba_train, y_train, model_name="", verbose=True):
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
        print(model_name + ' Train Accuracy: %.2f' % (train_accuracy * 100))
        print(model_name + ' Train Recall: %.2f' % (train_recall * 100))
        print(model_name + ' Train Precision: %.2f' % (train_precision * 100))
        print(model_name + ' Train F1: %.2f' % (train_f1 * 100))
        print(model_name + ' Test Accuracy: %.2f' % (test_accuracy * 100))
        print(model_name + ' Test Recall: %.2f' % (test_recall * 100))
        print(model_name + ' Test Precision: %.2f' % (test_precision * 100))
        print(model_name + ' Test F1: %.2f' % (test_f1 * 100))

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

    y_proba, y_proba_train = run_LSTM(X_train, y_train, X_test)
    results = evaluate(y_proba, y_test, y_proba_train, y_train, "LSTM")
    output.append(("LSTM", i, 'raw seq') + results)
    Y_proba.extend(y_proba)

    Y_targets.extend(y_test)
    Y_species.extend(y_test_species)
    i += 1

print("*******************SUMMARY*******************")

output_df = pd.DataFrame(output,
                         columns=['Model Name', 'Fold', 'Features', 'ap', 'auc', 'train_accuracy', 'train_recall',
                                  'train_precision', 'train_f1', 'test_accuracy', 'test_recall', 'test_precision',
                                  'test_f1'])
print(output_df)
output_df.to_csv('test_kuzmin_7_fold_CV_results.csv')

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
plt.savefig("test_kuzmin_ROC.jpg", dpi=400, bbox_inches="tight")
plt.clf()

# Generate pooled PR curve
plt.scatter(Y_proba, Y_species, facecolors='none', edgecolors='r')
plt.xlabel('Predicted probability')
plt.ylabel('Species')
plt.title('LSTM Predicted Proability vs. Infecting Species')
plt.savefig("test_kuzmin_PR.jpg", dpi=400, bbox_inches="tight")
plt.clf()
# -*- coding: utf-8 -*-
"""
Shared utilities for the mgm package.

@author: NBOLLIG
"""
import numpy as np
from tensorflow.keras.utils import to_categorical

def decode_from_one_hot(x, n_positions=60, n_characters=20):
    """
    Convert one-hot vector representation to integers. If all zeros, then returns -1 in that position.
    Expecting x in the shape of (n_positions, n_characters) but there is one extra gap character for the all-zero vector.
    """
    x = np.array(x).reshape((n_positions, n_characters))
    integers = np.argmax(x, axis=1, out=None).reshape(1, -1).tolist()[0]
    for i in range(x.shape[0]):
        if np.count_nonzero(x[i]) == 0:
            integers[i] = -1
    return integers

def encode_as_one_hot(x, n_positions=60, n_characters=20):
    """
    Convert array of integers to one-hot vector representation. If integer is -1, maps to all-zero vector.
    """
    x = np.array(x)
    if x.shape == (n_positions, n_characters):
        return x
    elif x.shape == (n_positions * n_characters,):
        return x.reshape((n_positions, n_characters))
    else:
        output = to_categorical(np.array(x), num_classes=n_characters).reshape(n_positions, n_characters)
        for i in range(n_positions):
            if x[i] == -1:
                output[i] = np.zeros((n_characters,))
        return np.array(output)

# def plot_aa_dist(pos, X_list, y_list, aa_vocab, class_label=None):
#     """
#     Plot the distribution of aa at a given position in the provided data.
#     Optional restriction by class_label.
#     """
#     def get_data(pos, X_list, y_list, aa_vocab, class_label):
#         X = np.concatenate(X_list)
#         y = np.concatenate(y_list)
#
#         indices = list(np.nonzero(y==class_label)[0])
#         X = X[indices, :, :]
#
#         dist = np.zeros((20,))
#
#         for i in range(X.shape[0]):
#             one_hot = X[i, pos]
#             aa_index = np.nonzero(one_hot == 1)[0].item()
#             dist[aa_index] += 1
#
#         dist = dist / np.sum(dist)
#         return dist
#
#     import matplotlib.pyplot as plt
#     if class_label == None:
#         dist_0 = get_data(pos, X_list, y_list, aa_vocab, class_label=0)
#         dist_1 = get_data(pos, X_list, y_list, aa_vocab, class_label=1)
#         plt.plot(aa_vocab, dist_0, label="Class 0")
#         plt.plot(aa_vocab, dist_1, label="Class 1")
#         plt.legend(loc='lower left')
#         plt.title("Class distributions")
#     else:
#         dist = get_data(pos, X_list, y_list, aa_vocab, class_label=class_label)
#         plt.plot(aa_vocab, dist)
#         plt.title("Distribution with class restriction = %i" % (class_label,))
#
#     plt.xlabel("Amino Acid")
#     plt.ylabel("Frequency at position %i" % (pos,))
    
    
    
        
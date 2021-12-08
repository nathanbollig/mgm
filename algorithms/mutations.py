# -*- coding: utf-8 -*-
"""
Functions to perform model-guided mutation to a sequence. The They will have the following inputs and outputs.

    input:
        seq - Sequence object or list of indices
        y - original label for the sequence
        aa_vocab - a list of amino acid characters in the indexed ordering (required if seq is a list of indices)
        model - a TensorFlow Model object
    output:
        a list of indices
        data - list of one dictionary per character flip

The aa_vocab may look like:
aa_vocab = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

@author: NBOLLIG
"""
import mgm
from mgm.common.utils import decode_from_one_hot, encode_as_one_hot
from mgm.common.sequence import Sequence
from mgm.algorithms.hotflip import one_hotflip
from mgm.algorithms.lookahead import one_lookahead_flip
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
import datetime

import tensorflow as tf

def derive_gradient_funct(model):
    """
    Older approach - plan to deprecate with update to TensorFlow v2.

    Return a gradient function that will be used to compute the gradient of the
    loss function of the model at x,y with respect to inputs.
    
    Parameters:
        model - keras model from big_bang
        x - single one-hot-encoded sequence
        y - binary label
    """
    
    # Set up function to compute gradient
    y_true = Input(shape=(1,))
    ce = K.binary_crossentropy(y_true, model.output)
    grad_ce = K.gradients(ce, model.inputs)
    return K.function(model.inputs + [y_true], grad_ce)

# TODO: create different method for randomized mgm
def greedy_mgm(seq, model=None, confidence_threshold = 0.5, type='hotflip', weights=None, gamma=0.1, cost=100, verbose=False):
    """
    Greedily iterate substitution mutations until the predicted class label flips and the
    resulting prediction has confidence >= confidence_threshold.

    input:
        seq - Sequence object
        model - a TensorFlow Model object
        confidence_threshold - mutation stops when model confidence exceeds this value (default: 0.5)
        type - method to select mutation at each step: 'hotflip' or 'lookahead_1'

    output:
        a list of indices
        data - list of one dictionary per character flip

    note:
        weights, gamma, cost - only used if type=='hotflip'

    """
    # Parameter validation
    if model == None:
        raise ValueError("Specify a model!")

    # Initial values
    y = seq.y
    pred = y
    conf = 0
    data = []
    init_pred_proba = model.predict(seq.to_predict()).item()
    i = 1

    while i < len(seq) and (int(y) == pred or conf < confidence_threshold):
        if verbose==True:
            print('.', end='', flush=True) # one dot per character flip

        time_start = datetime.datetime.now()

        # Execute one flip
        if type=='hotflip':
            seq, one_flip_data = one_hotflip(seq, model, weights=weights, gamma=gamma, cost=cost)
        if type=='lookahead_1':
            seq, one_flip_data = one_lookahead_flip(seq, model=model, verbose=verbose)

        # Apply model to updated sequence
        pred_proba = model.predict(seq.to_predict()).item()
        pred = int(pred_proba > 0.5)

        # Compute confidence that class label has been flipped
        if int(y) == 0:
            conf = pred_proba
        else:
            conf = 1 - pred_proba

        # Store values in flip data dictionary
        one_flip_data['pred_proba'] = pred_proba
        one_flip_data['conf'] = conf
        one_flip_data['init_pred_proba'] = init_pred_proba
        one_flip_data['change_number'] = i
        one_flip_data['time_sec'] = (datetime.datetime.now() - time_start).total_seconds()
        if seq.generator != None:
            one_flip_data['actual_label'] = seq.generator.predict(seq.integer_encoded)

        # Iterate
        data.append(one_flip_data)
        i += 1

    if verbose == True:
        print('')

    return seq, data


# def no_perturb(seq, y, aa_vocab, model, generator):
#     return seq, [{}]
#
#
# def random_pt_mutations(seq, y, aa_vocab, model, generator, k):
#     """
#     Mutate k randomly-selected amino acids, to a random distinct character.
#     """
#     index_list = random.sample(list(range(len(seq))), k)
#
#     for i in index_list:
#         candidates = [a for a in list(range(len(aa_vocab))) if a not in [i]]
#         j = random.choice(candidates)
#         seq[i] = j
#
#     return seq, [{}]
#
#
# def hot_flip(seq, y, aa_vocab, model):
#     """
#     Perform HotFlip algorithm once, i.e. - flip one character.
#     """
#     seq = encode_as_one_hot(seq)
#     seq, data = one_hotflip(model, seq, y)
#     return decode_from_one_hot(seq), [data]
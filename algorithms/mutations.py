# -*- coding: utf-8 -*-
"""
Functions to perform model-guided mutation to a sequence. The They will have the following inputs and outputs.

    input:
        seq - Sequence object or list of indices
        model - a TensorFlow Model object
    output:
        a Variant object

@author: NBOLLIG
"""
import mgm
from mgm.analysis.history import Variant, VariantList
from mgm.common.utils import decode_from_one_hot, encode_as_one_hot
from mgm.common.sequence import Sequence
from mgm.algorithms.hotflip import one_hotflip
from mgm.algorithms.lookahead import one_lookahead_flip
from mgm.algorithms.mgm import mgm_d
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


def variants(seq, model=None, N=10, confidence_threshold = 0.5, type='hotflip', weights=None, gamma=0.1, cost=100, verbose=False, fixed_iterations=None):
    """
    Run N independent variant searches.

    input:
        seq - Sequence object
        model - a TensorFlow Model object
        N - number of variants to produce
        confidence_threshold - mutation stops when model confidence exceeds this value (default: 0.5)
        type - method to select mutation at each step: 'hotflip' or 'lookahead_1'
        fixed_iterations - Specifies the maximum number of iterations for a given variant search. If set to an integer,
                            then stop after that number of iterations.

    output:
        VariantList object
    """
    variant_list = VariantList()

    for _ in range(N):
        variant = variant_search(seq=seq, model=model, confidence_threshold=confidence_threshold, type=type,
                                 weights=weights, gamma=gamma, cost=cost, verbose=verbose,
                                 fixed_iterations=fixed_iterations)

        variant_list.append(variant)

    return variant_list

def variant_search(seq, model=None, confidence_threshold = 0.5, type='hotflip', weights=None, gamma=0.1, cost=100, verbose=False, fixed_iterations=None, loss=None):
    """
    Iterate substitution mutations, using the designated selection strategy, until the predicted class label flips
    and the resulting prediction has confidence >= confidence_threshold. Apply to one sequence object.
    Outputs one Variant object (history of mutation trajectory).

    input:
        seq - Sequence object
        model - a TensorFlow Model object
        confidence_threshold - mutation stops when model confidence exceeds this value (default: 0.5)
        type - method to select mutation at each step: 'hotflip', 'lookahead_1', 'mgm-d'
        fixed_iterations - if set to an integer, then stop after that number of iterations
        loss - If true, the gradient is model loss wrt inputs. If false, the gradient is model output wrt inputs.

    output:
        a list of indices
        data - list of one dictionary per character flip

    note:
        weights, gamma, cost - only used if type=='hotflip'

    """
    # Parameter validation
    if model == None:
        raise ValueError("Specify a model!")

    # Copy the seq so original object is not changed
    seq = seq.copy()

    # Initial values
    y = seq.y
    pred = y
    conf = 0
    data = []
    init_seq = seq.copy()
    init_pred_proba = model.predict(seq.to_predict()).item()
    i = 1

    while i < len(seq) and (int(y) == pred or conf < confidence_threshold):
        if verbose==True:
            print('.', end='', flush=True) # one dot per character flip

        time_start = datetime.datetime.now()

        # Execute one flip
        if type=='hotflip':
            if loss is None:
                loss = True
            seq, one_flip_data = one_hotflip(seq, model, weights=weights, gamma=gamma, cost=cost, loss=loss)
        if type=='lookahead_1':
            seq, one_flip_data = one_lookahead_flip(seq, model=model, verbose=verbose)
        if type=='mgm-d':
            seq, one_flip_data = mgm_d(seq, init_seq, model=model, loss=loss)

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
        one_flip_data['change_number'] = i
        one_flip_data['time_sec'] = (datetime.datetime.now() - time_start).total_seconds()
        if seq.generator != None:
            one_flip_data['actual_label'] = seq.generator.predict(seq.integer_encoded)

        # Append data
        data.append(one_flip_data)

        # Stop after fixed iterations
        if fixed_iterations != None:
            if isinstance(fixed_iterations, int) == True:
                if i >= fixed_iterations:
                    break

        # Iterate
        i += 1



    # Create history object
    hx = Variant()
    hx.set_mgm_output(final_seq=seq, substitution_data=data)
    hx.set_init_seq(init_seq=init_seq)
    hx.set_fields(init_pred=init_pred_proba, confidence_threshold=confidence_threshold, algorithm_type=type) # TODO: pass any additional params passed to parent function

    # Compute variant cost
    hx.compute_cost("num_differences")

    if verbose == True:
        print('')

    return hx


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
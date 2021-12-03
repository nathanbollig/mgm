# -*- coding: utf-8 -*-
"""
Implementation of HotFlip approach to make a single character mutation.

@author: NBOLLIG
"""
import numpy as np
from mgm.algorithms.utils import compute_gradient

def one_hotflip(seq, model=None, ignore_char_indices=[-1], weights=None, gamma=0.1, cost=100):
    """
    Compute a single character flip using the HotFlip algorithm. Uses the one-hot encoding of the input sequence.
    
    Parameters:
        gradient_func - keras function for gradient computation on model
        seq - input sequence (a Sequence object)
        weights - if provided, then flip is randomized according to provided weights; it is a dict like MatrixInfo.blosum62
        gamma - provided if weights are provided
    
    Returns:
        Perturbed (one-hot-encoded) sequence
        Loss increase associated with the flip
    """
    # Get one-hot-encoded sequence
    x = seq.one_hot_encoded
    a_vector = seq.integer_encoded

    # get gradient
    output = compute_gradient(seq, model)
    
    if weights == None:
        # Find character flip that causes maximum increase in loss
        max_loss_increase = 0
        pos_to_change = None
        current_char_idx = None
        new_char_idx = None
        
        for i in range(seq.n_positions):
            a = a_vector[i]
            for b in range(seq.n_characters):
                if ignore_char_indices != None:
                    if a in ignore_char_indices:
                        continue
                loss_b = output[i][b]
                loss_a = output[i][a]
                loss_increase = loss_b - loss_a
                if loss_increase > max_loss_increase:
                    max_loss_increase = loss_increase
                    pos_to_change = i
                    current_char_idx = a
                    new_char_idx = b
    else:        
        assert(seq.aa_vocab != None)
        
        # Initialize distribution to sample from
        D = np.zeros((seq.n_positions, seq.n_characters))
        
        # Fill in with weighted loss increases
        for i in range(seq.n_positions):
            a = a_vector[i]
            for b in range(seq.n_characters):
                if ignore_char_indices != None:
                    if a in ignore_char_indices:
                        continue
                
                # Get estimated loss increase
                loss_b = output[i][b]
                loss_a = output[i][a]
                loss_increase = loss_b - loss_a
                
                # Get weight from blossum
                char_a = aa_vocab[a]
                char_b = aa_vocab[b]
                
                def score_match(pair, matrix):
                    if pair in matrix:
                        return matrix[pair]
                    elif (tuple(reversed(pair))) in matrix:
                        return matrix[(tuple(reversed(pair)))]
                    else:
                        return 0
                
                pair = (char_a, char_b)
                weight = np.exp(gamma * score_match(pair, weights))
                
                # Update D
                D[i][b] = weight * loss_increase
        
        # Apply cost to scale before softmax
        D_before = D
        D = cost * D     
        
        # Subtract max from D before applying softmax - does not change result but improves num stability
        D = D - np.max(D)
        
        # Softmax
        D = np.exp(D) / np.sum(np.exp(D))
        
        # Sample
        D_flattened = np.ravel(D)
        sample_idx = np.random.choice(len(D_flattened), p=D_flattened)
        pos_to_change, new_char_idx = np.unravel_index(sample_idx, D.shape)
        current_char_idx = a_vector[pos_to_change]
        max_loss_increase = output[pos_to_change][new_char_idx] - output[pos_to_change][current_char_idx]

    data = {}
    data['max_loss_increase'] = max_loss_increase
    data['pos_to_change'] = pos_to_change 
    data['current_char_idx'] = current_char_idx 
    data['new_char_idx'] = new_char_idx

    # Make substitution
    seq.sub(pos_to_change, new_char_idx)

    return seq, data

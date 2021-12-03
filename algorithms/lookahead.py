# -*- coding: utf-8 -*-
"""
Lookahead-based mutation approaches.

@author: NBOLLIG
"""

import numpy as np

def one_lookahead_flip(seq, model, ignore_char_indices=[-1], verbose=False):
    """
    Exhaustively search over all possible character substitutions at each position, and make the single substitution
    that causes greatest increase in model loss.

    Uses the one-hot encoding of the input sequence.
    """

    # Character sequence for x
    x = seq.one_hot_encoded
    a_vector = seq.integer_encoded

    # Find character flip that causes maximum increase in loss
    pos_to_change = None
    current_char_idx = None
    new_char_idx = None
    conf_start = model.predict(seq.to_predict()).item()
    candidate_seqs = []  # will store list of candidate sequences
    candidate_muts = []  # parallel list of tuples of form (position, current char idx, new char idx)

    for i in range(seq.n_positions):
        a = a_vector[i]
        if verbose:
            if i % 10 == 0:
                print(i, end=' ', flush=True)

        for b in range(seq.n_characters):
            # Pass through if character should be ignored
            if ignore_char_indices != None:
                if a in ignore_char_indices:
                    continue

            # Consider this candidate
            candidate_seq = np.copy(x).reshape((seq.n_positions, seq.n_characters))
            candidate_seq[i][a] = 0
            candidate_seq[i][b] = 1
            candidate_seqs.append(candidate_seq)
            candidate_muts.append((i, a, b))


    # Select candidate
    candidate_seqs = np.array(candidate_seqs).reshape(-1, seq.n_positions, seq.n_characters)
    preds = model.predict(candidate_seqs)
    preds = preds.ravel()
    j = np.argmax(preds)

    # Store mutation info
    mut = candidate_muts[j]
    pos_to_change, current_char_idx, new_char_idx = mut
    conf_end = preds[j]
    max_conf_increase = conf_end - conf_start

    data = {}
    data['max_loss_increase'] = max_conf_increase
    data['pos_to_change'] = pos_to_change
    data['current_char_idx'] = current_char_idx
    data['new_char_idx'] = new_char_idx

    # Make substitution
    seq.sub(pos_to_change, new_char_idx)

    return seq, data
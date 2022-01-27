import numpy as np
from mgm.algorithms.utils import compute_gradient

def mgm_d(seq, init_seq, model=None, representation='one-hot', cost_function='squared_difference', skip_indices=[-1], invalid_sub_indices=[-1], lambda_param=1e9, loss=False, hashes=None):
    """
    Compute a single character flip using the mgm_d algorithm, based on a fixed-length vector embedding for each
    amino acid.

    Parameters:
        seq - current sequence (a Sequence object)
        init_seq - initial sequence
        model - TensorFlow SavedModel object
        representation - Representation type being used ('one-hot', etc. - must be a key in seq.representation_space)
        cost_function - Cost function 'squared_difference', etc.
        skip_indices - do not substitute at positions where the character is in this list (list of indices)
        invalid_sub_indices - do not substitute with these characters (list of indices)
        lambda_param - hyperparameter in mgm-d
        loss - If true, the gradient is model loss wrt inputs. If false, the gradient is model output wrt inputs.
        hashes - Set object of hash values for all sequences seen so far.

    Returns:
        Variant object
        Data dictionary

    Given a current sequence $x'$ and initial sequence $x^{(0)}$, this searches over positions $i$ and new characters $b$ to find the
    one that maximizes the objective function

    \[\sum_j (x'_{ij} - x^{(0)}_{ij})(r_{bj} - x'_{ij}) - \lambda \sum_j \frac{\partial f(x)}{\partial x_{ij}}\rvert_{x=x'} \cdot (r_{bj}-x'_{ij})\]

    where $x_{ij}$ is the value at the $j$th coordinate of the representation of the amino acid at the $i$th position of $x$,
    and $r_{bj}$ is the $j$th component of the amino acid corresponding to index $b$, i.e. the $b$th row of the
    representation space $R$ for this representation.
    """
    # Get encoded sequence
    if representation == 'one-hot':
        x_current = seq.one_hot_encoded
        x_init = init_seq.one_hot_encoded
    else:
        x_current = seq.get_encoding(representation)
        x_init = init_seq.get_encoding(representation)

    # Get integer-encoded sequence
    a_vector = seq.integer_encoded

    # Get representation space
    R = seq.representation_space[representation]

    # Get gradient
    output = compute_gradient(seq, model, loss=loss)

    # Find character flip that maximizes objective function
    min_objective_value = None
    pos_to_change = None
    current_char_idx = None
    new_char_idx = None

    for i in range(seq.n_positions):
        a = a_vector[i]

        # Skip substitution at designated indices
        if skip_indices != None:
            if a in skip_indices:
                continue

        for b in range(seq.n_characters):
            # Skip substitution with designated indices
            if invalid_sub_indices != None:
                if b in invalid_sub_indices:
                    continue

            # Compute array of term1 values over components of representation
            term1_factor1 = x_current[i] - x_init[i]
            term1_factor2 = R[b] - x_current[i]
            term1 = np.multiply(term1_factor1, term1_factor2)

            # Compute array of term2 values over components of representation
            term2_factor1 = output[i]
            term2_factor2 = R[b] - x_current[i]
            term2 = np.multiply(term2_factor1, term2_factor2)

            # Combine terms into array of objective values over positions
            objective_array = term1 - lambda_param * term2

            # Sum over positions
            objective_value = np.sum(objective_array)

            # Update if new minimum was found
            if min_objective_value is None:
                min_objective_value = objective_value
            elif objective_value < min_objective_value:
                if hashes is not None and seq.get_hash_of_sub(i, b) in hashes:
                    continue  # don't accept mutation that has been seen before
                min_objective_value = objective_value
                pos_to_change = i
                current_char_idx = a
                new_char_idx = b

    # Make substitution
    seq.sub(pos_to_change, new_char_idx)
    if hashes is not None:
        hashes.add(seq.get_hash())

    data = {}
    data['min_objective_value'] = min_objective_value
    data['pos_to_change'] = pos_to_change
    data['current_char_idx'] = current_char_idx
    data['new_char_idx'] = new_char_idx

    return seq, data
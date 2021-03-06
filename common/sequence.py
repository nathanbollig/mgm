"""
Represent a single amino acid sequence.
"""
from mgm.common.utils import decode_from_one_hot, encode_as_one_hot
import numpy as np
import pickle
from mgm.common.utils import get_full_path

def compute_hash(integer_encoded):
    """
    Produce a hash based on this sequence. Based on Python hash of a string formed from the integer encoding.
    """
    int_string = ','.join(str(e) for e in integer_encoded)
    return hash(int_string)

def get_kidera_factors():
    """
    Load in pickle file and return dictionary of format dict[char] = vector.
    """
    input_file_name = get_full_path("common", "map_attribute.pkl")
    with open(input_file_name, "rb") as f:
        map_attribute = pickle.load(f)

    # Convert values to numpy arrays
    for key, val in map_attribute.items():
        map_attribute[key] = np.array(val)

    return map_attribute

def mult_align_idx_to_unaligned_idx(integer_encoded_seq, i):
    """
    Input:
        integer_encoded_seq - 1d array with -1 representing gap
        i - index in multiple seq alignment

    Output:
        Index in sequence with gap chars removed
    """
    if i >= len(integer_encoded_seq):
        raise IndexError("Index %i exceeds the length of the input sequence (%s)" % (i, len(integer_encoded_seq)))

    unique, counts = np.unique(integer_encoded_seq[:i+1], return_counts=True)
    num_gaps = dict(zip(unique, counts)).get(-1)
    if num_gaps is None:
        num_gaps = 0
    return max(0, i - num_gaps)

def unaligned_idx_to_mult_align_idx(integer_encoded_seq, i):
    """
    Input:
        integer_encoded_seq - 1d array with -1 representing gap
        i - index in sequence with gap characters removed

    Output:
        Index in mult seq alignment
    """
    integer_encoded_seq = np.array(integer_encoded_seq)
    non_gap_indices = np.nonzero(integer_encoded_seq != -1)[0]

    if i >= len(non_gap_indices):
        raise IndexError("Index %i exceeds the length of the unaligned input sequence (%s)" % (i, len(non_gap_indices)))

    return non_gap_indices[i]

# def encoded_to_integer(x, R):
#     """
#     Convert an encoded sequence to integer representation.
#
#     Inputs:
#         x - 2D encoded representation or 3D array of encoded representations
#         R - 2D array of dimension n_characters x representation dimension
#
#     e.g. for x a single, Kidera-encoded sequence, it has shape (n_positions, 10) and R would have dimension n_characters x 10
#
#     Output:
#         Returns the integer-encoded representation
#     """
#
#     def vec_to_int(vec, R):
#         if np.all((vec == 0)):
#             return -1
#         else:
#             return np.where(np.all(R == vec, axis=1))[0].item()  # Fails because assumes rows are unique but C and U are the same (2 and 19 in current aa_vocab)
#
#     def encoded_to_int_2d(x, R):
#         x_int = []
#         for i in range(x.shape[0]):
#             vec = x[i]
#             x_int.append(vec_to_int(vec, R))
#         return np.array(x_int)
#
#     if len(x.shape) == 2:  # single sequence
#         return encoded_to_int_2d(x, R)
#
#     elif len(x.shape) == 3:  # array of sequences
#         x_int = []
#         for item in range(x.shape[0]):
#             x_int.append(encoded_to_int_2d(x[item], R))
#         return np.array(x_int)


class Sequence:
    def __init__(self, x, y=None, aa_vocab=None, n_positions=None, n_characters=None, generator=None):
        """
        Create a Sequence object

        Input:
            x - a sequence represented as a list or 1d array (integer-encoded), or 2d array (one-hot-encoded)
            y - label
            aa_vocab - a list of amino acid characters in the indexed ordering (required if seq is a list of indices)
            n_positions - number of positions in the sequence
            n_characters - size of the character alphabet (must match length of aa_vocab)
            generator - may use to track a pointer to the generator object for this sequence

        The aa_vocab may look like:
        aa_vocab = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

        The representation_space is a dictionary of format representation_space[type] = matrix, where
        the matrix is n_characters x amino acid embedding dimension. This stores information regarding fixed-length
        amino acid embeddings.
        """

        # Class fields set by the constructor
        self.x = x
        self.y = y
        self.n_positions = n_positions
        self.n_characters = n_characters
        self.aa_vocab = aa_vocab
        self.generator = generator
        self.integer_encoded = None
        self.one_hot_encoded = None
        self.representation_space = {}

        # Verify aa_vocab length matches number of characters
        if aa_vocab != None:
            if n_characters != None:
                if n_characters != len(aa_vocab):
                    raise ValueError("n_characters must match the length of aa_vocab.")
            else:
                self.n_characters = len(aa_vocab) # can infer n_characters from aa_vocab if necessary

            # Convert vocab to upper case
            aa_vocab = [char.upper() for char in aa_vocab]
            self.aa_vocab = aa_vocab

        # Verify required arguments
        if self.n_characters == None:
            raise ValueError("Must specify n_characters or aa_vocab.")

        # CASE: x is a 1d array (assume equivalent to integer-encoded list)
        if isinstance(x, np.ndarray) and len(x.shape) == 1:
            x = x.tolist()

        # CASE: x is a list of non-integers => convert to list of ints
        if type(x) == list and type(x[0]) != int:
            x = [int(i) for i in x]

        # CASE: x is a integer-encoded list
        if type(x) == list and type(x[0]) == int:
            # Store as integer-encoded sequence
            self.integer_encoded = x

            # Infer n_positions
            if n_positions != None:
                if len(x) != n_positions:
                    raise ValueError("n_positions does not match length of x, which is inferred to be an integer-encoded list.")
            else:
                self.n_positions = len(x)

            # Store one-hot-encoded sequence
            self.one_hot_encoded = encode_as_one_hot(x, n_positions=self.n_positions, n_characters=self.n_characters)

        # CASE: x is a 2d array (assume one-hot-encoded)
        if isinstance(x, np.ndarray) and len(x.shape) == 2 and x.shape[1] == self.n_characters:
            # Store as one-hot sequence
            self.one_hot_encoded = x

            # Infer n_positions
            if n_positions != None:
                if x.shape[0] != n_positions:
                    raise ValueError(
                        "n_positions does not match length of x, which is inferred to be one-hot encoded.")
            else:
                self.n_positions = x.shape[0]

            # Store integer-encoded sequence
            self.integer_encoded = decode_from_one_hot(x, n_positions=self.n_positions, n_characters=self.n_characters)

        # Enforce formatting
        self.one_hot_encoded = self.one_hot_encoded.astype('float32')
        self.integer_encoded = np.array(self.integer_encoded, dtype='int32')

        # Set one-hot matrix in representation space
        self.representation_space['one-hot'] = np.identity(self.n_characters)

        # Set kidera representation space
        if aa_vocab is not None:
            map_attribute = get_kidera_factors()  # Get Kidera representations from Ze
            non_ambiguous_chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                                   'V', 'W', 'Y']
            R_kidera = []
            for char in aa_vocab:
                if char == 'X':
                    vals = [map_attribute[c] for c in non_ambiguous_chars]
                    vector = sum(vals) / len(vals)
                elif char == 'Z':
                    vector = (map_attribute['E'] + map_attribute['Q']) / 2.0  # glutamate or glutamine
                elif char == 'B':
                    vector = (map_attribute['D'] + map_attribute['N']) / 2.0  # aspartate or asparagine
                elif char == 'J':
                    vector = (map_attribute['L'] + map_attribute['I']) / 2.0  # leucine or isoleucine
                elif char == 'U':
                    vector = map_attribute['C']  # Treat selenocysteine as cysteine
                else:
                    vector = map_attribute[char]

                R_kidera.append(vector)

            self.representation_space['kidera'] = np.array(R_kidera).astype(np.float32)

    def get_encoding(self, type):
        """
        Return encoding of sequence. Type param is key into representation_space.
        """
        R = self.representation_space[type]
        return np.matmul(self.one_hot_encoded, R)

    def sub(self, pos_to_change, new_char_idx):
        self.integer_encoded[pos_to_change] = new_char_idx
        self.one_hot_encoded = encode_as_one_hot(self.integer_encoded, n_positions=self.n_positions, n_characters=self.n_characters)
        self.one_hot_encoded.astype('float32')

    def to_predict(self, representation='one_hot'):
        """
        Returns the sequence in the appropriate format and shape for passing into model prediction.

        Input:
            representation - indicates the desired representation ('one-hot')
        """

        if representation == 'one_hot':
            return self.one_hot_encoded.reshape(1, self.n_positions, self.n_characters)
        else:
            return self.get_encoding(representation).reshape(1, self.n_positions, -1)

    def __len__(self):
        return self.n_positions

    def copy(self):
        x = self.integer_encoded.copy()
        aa_vocab, generator = None, None

        if self.aa_vocab != None:
            aa_vocab = self.aa_vocab.copy()
        if self.generator != None:
            generator= self.generator.copy()

        new = Sequence(x=x, y=self.y, aa_vocab=aa_vocab, n_positions=self.n_positions, n_characters=self.n_characters, generator=generator)

        if hasattr(self, 'species'):
            new.set_species(self.species)

        if hasattr(self, 'defline'):
            new.set_defline(self.defline)

        return new

    def get_hash_of_sub(self, pos_to_change, new_char_idx):
        new_integer_encoded = self.integer_encoded.copy()
        new_integer_encoded[pos_to_change] = new_char_idx
        return compute_hash(new_integer_encoded)

    def get_hash(self):
        return compute_hash(self.integer_encoded)

    def set_species(self, species):
        self.species = species

    def get_species(self):
        if hasattr(self, 'species'):
            return self.species
        else:
            return None

    def set_defline(self, defline):
        self.defline = defline

    def get_defline(self):
        if hasattr(self, 'defline'):
            return self.defline
        else:
            return None
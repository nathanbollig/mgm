"""
Represent a single amino acid sequence.
"""
from mgm.common.utils import decode_from_one_hot, encode_as_one_hot
import numpy as np

def compute_hash(integer_encoded):
    """
    Produce a hash based on this sequence. Based on Python hash of a string formed from the integer encoding.
    """
    int_string = ','.join(str(e) for e in integer_encoded)
    return hash(int_string)

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
        if isinstance(x, np.ndarray) and len(x.shape) == 2:
            # Store as one-hot sequence
            self.one_hot_encoded = x

            # Infer n_positions
            if n_positions != None:
                if x.shape[0] != n_positions:
                    raise ValueError(
                        "n_positions does not match length of x, which is inferred to be an integer-encoded list.")
            else:
                self.n_positions = x.shape[0]

            # Store integer-encoded sequence
            self.integer_encoded = decode_from_one_hot(x, n_positions=self.n_positions, n_characters=self.n_characters)

        # Enforce formatting
        self.one_hot_encoded = self.one_hot_encoded.astype('float32')
        self.integer_encoded = np.array(self.integer_encoded, dtype='int32')

        # Set one-hot matrix in representation space
        self.representation_space['one-hot'] = np.identity(self.n_characters)

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
            return None

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
        return new

    def get_hash_of_sub(self, pos_to_change, new_char_idx):
        new_integer_encoded = self.integer_encoded.copy()
        new_integer_encoded[pos_to_change] = new_char_idx
        return compute_hash(new_integer_encoded)

    def get_hash(self):
        return compute_hash(self.integer_encoded)

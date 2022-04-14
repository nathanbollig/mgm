"""
Variant class to be returned by a mutation algorithm. Represents history for a trajectory of mutations starting from a
single initial sequence.

Fields can be set by set_fields(**kwargs) which depends on the params passed by the caller.

Available fields:
    final_seq - Sequence object corresponding to the output of the mutation algorithm

    initial_seq - Sequence object corresponding to the input to the mutation algorithm

    substitution_data - DataFrame of data for each character substitution:
            max_loss_increase, pos_to_change, current_char_idx, new_char_idx,
            pred_proba, conf, change_number, time_sec, etc.

    init_pred - initial prediction probability of the original sequence

    confidence_threshold - threshold used as part of mgm stopping condition

    algorithm_type - type of mgm algorithm performed

    variant_cost - cost of the variant under the designated variant cost function

    variant_cost_type - cost function used to compute variant cost (e.g. "squared_difference")

"""
from mgm.common.cost_functions import squared_difference, num_differences

class Variant:
    def __init__(self):
        self.final_seq = None
        self.init_seq = None
        self.substitution_data = None
        self.init_pred = None
        self.variant_cost = None
        self.variant_cost_type = None
        self.variant_risk = None
        self.variant_risk_type = None

    def set_mgm_output(self, final_seq,substitution_data):
        self.final_seq = final_seq
        self.substitution_data = substitution_data

    def set_init_seq(self, init_seq):
        self.init_seq = init_seq

    def set_fields(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)

    def compute_cost(self, type="squared_difference"):
        if self.fixed_iterations == len(self.substitution_data):  # The mutation trajectory was truncated by fixed iteration limit
            self.variant_cost = "undefined"
            self.variant_cost_type = "n/a"
        elif type=="num_differences":
            self.variant_cost = num_differences(self.init_seq, self.final_seq)
            self.variant_cost_type = "num_differences"
        elif type=="squared_difference":
            self.variant_cost = squared_difference(self.init_seq, self.final_seq, representation=self.representation)
            self.variant_cost_type = "squared_difference"

        return self.variant_cost

    def compute_risk(self, type):
        if type=="reciprocate_cost":
            self.variant_risk = 1/(self.variant_cost + 1)
            self.variant_risk_type = "reciprocate_cost"
        return self.variant_risk

    def is_same_trajectory(self, other_variant):
        # Check lengths are the same
        if len(self.substitution_data) != len(other_variant.substitution_data):
            return False

        # Check pos to change and new chars are the same
        for i in range(len(self.substitution_data)):
            sub_data = self.substitution_data[i]
            other_sub_data = other_variant.substitution_data[i]
            if sub_data['pos_to_change'] != other_sub_data['pos_to_change']:
                return False
            if sub_data['new_char_idx'] != other_sub_data['new_char_idx']:
                return False

        return True

    def get_final_pred(self):
        return self.substitution_data[-1]['conf']

    def replay_trajectory(self):
        seq = self.init_seq.copy()
        for data in self.substitution_data:
            i = data['pos_to_change']
            b = data['new_char_idx']
            seq.sub(i, b)
        return seq

class VariantList:
    def __init__(self):
        self.variants = []

    def append(self, variant):
        assert(isinstance(variant, Variant) == True)
        self.variants.append(variant)

    def initial_pred(self, i=0):
        """
        Retrieve the initial prediction stored in the ith Variant, which by default is the first.
        """
        if len(self) > 0:
            return self.variants[i].init_pred
        else:
            return None

    def __len__(self):
        return len(self.variants)

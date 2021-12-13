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

    def set_mgm_output(self, final_seq,substitution_data):
        self.final_seq = final_seq
        self.substitution_data = substitution_data

    def set_init_seq(self, init_seq):
        self.init_seq = init_seq

    def set_fields(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)

    def compute_cost(self, type):
        if type=="num_differences":
            self.variant_cost = num_differences(self.init_seq, self.final_seq)
            self.variant_cost_type = "num_differences"
        if type=="squared_difference":
            self.variant_cost = squared_difference(self.init_seq, self.final_seq)
            self.variant_cost_type = "squared_difference"

        return self.variant_cost
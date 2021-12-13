"""
Functions for measuring the risk of a Variant or VariantSet
"""
import numpy as np

def risk_of_variant(variant):
    cost = variant.variant_cost
    return _cost_to_risk(cost)

def _cost_to_risk(cost):
    return np.exp(-cost)
import pickle

import pandas as pd

from mgm.common.sequence import unaligned_idx_to_mult_align_idx
from mgm.common.utils import set_data_directory
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# Set analysis parameters
########################################################################################################################
data_dir = "spillover_simulation9"
SPILL_SEQ_DEFLINE = 'RaTG13|QHR63300|Bat|SARS_CoV_2'
SPILL_SEQ_PRETTY = 'RaTG13'
WITHHELD_SPECIES = 'SARS_CoV_2'
WITHHELD_SPECIES_PRETTY = 'SARS CoV 2'
########################################################################################################################

# Set directory to where results are
set_data_directory(data_dir)

# Load variants
with open(r"variants.pkl", "rb") as f:
    variants = pickle.load(f)

# Correction for spillover_simulation9
def truncate_mutation_trajectory(substitution_data, confidence_threshold):
    for i, sub_dict in enumerate(substitution_data):
        if sub_dict['pred_proba'] > confidence_threshold:
            substitution_data_truncated = substitution_data[:i+1]
            return substitution_data_truncated
    return substitution_data

for variant in variants:
    final_pred = variant.substitution_data[-1]['conf']
    variant.confidence_threshold = 0.95
    variant.substitution_data = truncate_mutation_trajectory(variant.substitution_data, variant.confidence_threshold)
    variant.compute_cost("num_differences")

#analyze_variants(variants, filename="rankings_corrected.csv")

# Pull out thresholds and auc for each method
thresh_avg_species = []
thresh_all_seq = []
auc_avg_species = []
auc_all_seq = []

for variant in variants:
    data = variant.LOOCV_data
    thresh_avg_species.append(data['threshold_based_on_avg_within_species'])
    thresh_all_seq.append(data['threshold_based_on_all_seq'])
    auc_avg_species.append(data['auc_based_on_avg_within_species'])
    auc_all_seq.append(data['auc_based_on_all_seq'])

# AUC chart
plt.scatter(auc_avg_species, auc_all_seq, edgecolors='green', facecolors='none', alpha=0.7)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("LOOCV AUROC - using mean model score within viral species")
plt.ylabel("LOOCV AUROC - using model score on all sequences")
plt.title("Model predictive performance using two pooling approaches")
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 0.5, linestyle='--')
plt.savefig("AUC_chart.jpg", dpi=400)

# Threshold chart
plt.scatter(thresh_avg_species, thresh_all_seq, edgecolors='green', facecolors='none', alpha=0.7)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Selected threshold - using mean model score within viral species")
plt.ylabel("Selected threshold - using model score on all sequences")
plt.title("Threshold selected for MGM-d using two pooling approaches")
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 0.5, linestyle='--')
plt.savefig("threshold_chart.jpg", dpi=400)

# Where are SARS CoV 2 mutations made?
def get_positions(variants, confidence_threshold=0.95):
    positions = []
    spill_seq_positions = set()
    for variant in variants:
        position_set = set()  # Keep set of positions within each variant, i.e. only count 1 mutation per position per sequence at indicated conf threshold
        if variant.init_seq.species == WITHHELD_SPECIES:
            for i, sub_dict in enumerate(variant.substitution_data):
                if sub_dict['pred_proba'] <= confidence_threshold:
                    index = sub_dict['pos_to_change']
                    if variant.init_seq.defline == SPILL_SEQ_DEFLINE:
                        spill_seq_positions.add(index)
                    else:
                        position_set.add(index)
        positions.extend(list(position_set))
    return positions, list(spill_seq_positions)

positions95, spill_seq_positions95 = get_positions(variants, confidence_threshold=0.95)
positions90, spill_seq_positions90 = get_positions(variants, confidence_threshold=0.90)
positions85, spill_seq_positions85 = get_positions(variants, confidence_threshold=0.85)
positions80, spill_seq_positions80 = get_positions(variants, confidence_threshold=0.80)
positions75, spill_seq_positions75 = get_positions(variants, confidence_threshold=0.75)

plt.clf()
bins = np.linspace(0, 2396, 90)
_, _, _ = plt.hist(positions95, bins, density=False, facecolor='y', edgecolor='k', alpha=0.8, label='conf=.95')
_, _, _ = plt.hist(positions85, bins, density=False, facecolor='b', edgecolor='k', alpha=0.8, label='conf=.85')
_, _, patches = plt.hist(positions75, bins, density=False, facecolor='r', edgecolor='k', alpha=0.8, label='conf=.75')
plt.xlabel('Position')
plt.ylabel('Count')

spill_seq_positions_bins = np.digitize(spill_seq_positions75, bins)
for i, bin_idx in enumerate(spill_seq_positions_bins):
    # patches[i].set_linestyle('--')
    # patches[i].set_linewidth(1.5)
    # patches[i].set_edgecolor('black')
    patch = patches[bin_idx]
    x_center = patch.get_x() + 0.5 * patch.get_width()
    plt.scatter(x=x_center, y=3, marker="*", linestyle="--", color="black", label="%s at conf=.75" % (SPILL_SEQ_PRETTY,) if i==0 else "")

plt.legend(fontsize='small')
plt.title('Mutations suggested in %s' % (WITHHELD_SPECIES_PRETTY,))
plt.savefig('distribution.jpg', dpi=400)

# Finding regions
def SARSCoV2_endpoints(variants):
    # First get representative variant
    for i,variant in enumerate(variants):
        if variant.init_seq.defline == SPILL_SEQ_DEFLINE:
            idx = i
    # Then use it to get desired cutoffs
    """
    Li, Fang, et al. "Structure of SARS coronavirus spike receptor-binding domain complexed with receptor." Science 309.5742 (2005): 1864-1868.
    +
    https://www.nature.com/articles/nrmicro2090
    """
    seq = variants[idx].init_seq.integer_encoded
    def get_endpoints(a, b):
        return unaligned_idx_to_mult_align_idx(seq, a), unaligned_idx_to_mult_align_idx(seq, b)

    endpoints = {}
    endpoints['SP'] = get_endpoints(1, 13)
    endpoints['S1'] = get_endpoints(13, 770)
    endpoints['S2'] = get_endpoints(667, 1190)
    endpoints['RBD'] = get_endpoints(318, 510)
    endpoints['RBM'] = get_endpoints(424, 494)
    endpoints['FP'] = get_endpoints(770, 788)
    endpoints['HR1'] = get_endpoints(892, 1013)
    endpoints['HR2'] = get_endpoints(1145, 1195)
    endpoints['TM'] = get_endpoints(1195, 1215)
    endpoints['CP'] = get_endpoints(1215, 1255)
    pd.DataFrame(endpoints).to_csv("motif_endpoints.csv")

if WITHHELD_SPECIES = 'SARS_CoV_2':
    SARSCoV2_endpoints(variants)

# Comparison of rankings by MGM vs init model pred - 2D
rankings = pd.read_csv('rankings_corrected.csv')
rankings = rankings.loc[rankings['Cost'] != 'undefined']
rankings['MGM_rank'] = range(1, len(rankings) + 1)
rankings = rankings.sort_values(by=['Initial pred'], ascending=False)
rankings['model_rank'] = range(1, len(rankings) + 1)
rankings['rank_change'] = rankings['model_rank'] - rankings['MGM_rank']
rankings.to_csv('rankings_corrected_with_ranks.csv')

rankings1 = rankings.loc[rankings['Species'] == WITHHELD_SPECIES]
rankings0 = rankings.loc[rankings['Species'] != WITHHELD_SPECIES]
rankings_spill_seq = rankings.loc[rankings['defline'] == SPILL_SEQ_DEFLINE]

plt.clf()
plt.scatter(rankings1['model_rank'], rankings1['MGM_rank'], s=50, facecolors='none', edgecolors='r', label=WITHHELD_SPECIES_PRETTY)
plt.scatter(rankings0['model_rank'], rankings0['MGM_rank'], s=50, facecolors='none', edgecolors='b', label='Other groups')
plt.scatter(rankings_spill_seq['model_rank'], rankings_spill_seq['MGM_rank'], s=15, facecolors='black', edgecolors='black', marker="*", label=SPILL_SEQ_PRETTY)
plt.xlabel('Risk ranking by initial model score')
plt.ylabel('Risk ranking by MGM-d')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize='small')
plt.title('Comparison of ranking methods')
m = len(rankings)+1
plt.xlim(1,m)
plt.ylim(1,m)
plt.plot([1, m], [1, m], color = 'black', linewidth = 0.5, linestyle='--')
plt.savefig('rank_scatter.jpg', dpi=400, bbox_inches="tight")

# As above, ranking change in 1D
plt.clf()
change1 = rankings1['rank_change'].to_list()
change0 = rankings0['rank_change'].to_list()
spill_seq_profit = rankings_spill_seq['rank_change'].to_list()
lower_limit = min(change0 + change1 + spill_seq_profit)
upper_limit = max(change0 + change1 + spill_seq_profit)
bins = np.linspace(lower_limit, upper_limit, 50)

plt.axvline(x=0, linestyle="--", color="black")
plt.hist(change0, bins, density=False, facecolor='b', edgecolor='k', alpha=0.8, label='Other groups')
plt.hist(change1, bins, density=False, facecolor='r', edgecolor='k', alpha=0.8, label=WITHHELD_SPECIES_PRETTY)
plt.scatter(x=spill_seq_profit, y = 1, marker="*", label="RaTG13", linestyle="--", color="black")

plt.xlabel('Relative increase in ranking via MGM')
plt.ylabel('Count')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.title('Ranking change relative to ranking by model score')
plt.savefig('rank_change_distribution.jpg', dpi=400)




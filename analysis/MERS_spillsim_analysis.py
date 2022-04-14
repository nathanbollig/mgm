import pickle

import pandas as pd

from mgm.analysis.SARSCoV2_spillsim_analysis import rankings_spill_seq
from mgm.analysis.trajectory_analysis import conf_vs_change_number
from mgm.common.sequence import unaligned_idx_to_mult_align_idx
from mgm.common.utils import set_data_directory
import matplotlib.pyplot as plt
import numpy as np
from mgm.pipelines.spillover_simulation import analyze_variants, reanalyze_variants

########################################################################################################################
# Set analysis parameters
########################################################################################################################
data_dir = "spillover_simulation_MERS"
WITHHELD_SPECIES = 'Middle_East_respiratory_syndrome_coronavirus'
WITHHELD_SPECIES_PRETTY = 'MERS'
THRESHOLD = 0.04
keep_final_seq = False
########################################################################################################################

if THRESHOLD is None:
    suffix = ''
else:
    assert (THRESHOLD <= 1)
    suffix = '_' + str(int(THRESHOLD * 100))

rankings_path = 'rankings%s.csv' % (suffix,)

if keep_final_seq == True:
    suffix = suffix + "_keepfinal"

# Set directory to where results are
set_data_directory(data_dir)

# Load variants
with open(r"variants.pkl", "rb") as f:
    variants = pickle.load(f)

# Make confidence trajectories
conf_vs_change_number(variants)

if THRESHOLD is not None:
    reanalyze_variants(variants, THRESHOLD, rankings_path, keep_final_seq=keep_final_seq)

# # Pull out thresholds and auc for each method
# thresh_avg_species = []
# thresh_all_seq = []
# auc_avg_species = []
# auc_all_seq = []
#
# for variant in variants:
#     data = variant.LOOCV_data
#     thresh_avg_species.append(data['threshold_based_on_avg_within_species'])
#     thresh_all_seq.append(data['threshold_based_on_all_seq'])
#     auc_avg_species.append(data['auc_based_on_avg_within_species'])
#     auc_all_seq.append(data['auc_based_on_all_seq'])
#
# # AUC chart
# plt.scatter(auc_avg_species, auc_all_seq, edgecolors='green', facecolors='none', alpha=0.7)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xlabel("LOOCV AUROC - using mean model score within viral species")
# plt.ylabel("LOOCV AUROC - using model score on all sequences")
# plt.title("Model predictive performance using two pooling approaches")
# plt.plot([0, 1], [0, 1], color = 'black', linewidth = 0.5, linestyle='--')
# plt.savefig("AUC_chart.jpg", dpi=400)
#
# # Threshold chart
# plt.scatter(thresh_avg_species, thresh_all_seq, edgecolors='green', facecolors='none', alpha=0.7)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xlabel("Selected threshold - using mean model score within viral species")
# plt.ylabel("Selected threshold - using model score on all sequences")
# plt.title("Threshold selected for MGM-d using two pooling approaches")
# plt.plot([0, 1], [0, 1], color = 'black', linewidth = 0.5, linestyle='--')
# plt.savefig("threshold_chart.jpg", dpi=400)

# Where are mutations made?
def get_positions(variants, confidence_threshold=0.95):
    positions = []
    spill_seq_positions = set()
    for variant in variants:
        position_set = set()  # Keep set of positions within each variant, i.e. only count 1 mutation per position per sequence at indicated conf threshold
        if variant.init_seq.species == WITHHELD_SPECIES:
            for i, sub_dict in enumerate(variant.substitution_data):
                if sub_dict['pred_proba'] <= confidence_threshold:
                    index = sub_dict['pos_to_change']
                    position_set.add(index)
        positions.extend(list(position_set))
    return positions

positions95 = get_positions(variants, confidence_threshold=0.95)
positions90 = get_positions(variants, confidence_threshold=0.90)
positions85 = get_positions(variants, confidence_threshold=0.85)
positions80 = get_positions(variants, confidence_threshold=0.80)
positions75 = get_positions(variants, confidence_threshold=0.75)

plt.clf()
bins = np.linspace(0, 2396, 90)
_, _, _ = plt.hist(positions95, bins, density=False, facecolor='y', edgecolor='k', alpha=0.8, label='conf=.95')
_, _, _ = plt.hist(positions85, bins, density=False, facecolor='b', edgecolor='k', alpha=0.8, label='conf=.85')
_, _, patches = plt.hist(positions75, bins, density=False, facecolor='r', edgecolor='k', alpha=0.8, label='conf=.75')
plt.xlabel('Position')
plt.ylabel('Count')

plt.legend(fontsize='small')
plt.title('Mutations suggested in %s' % (WITHHELD_SPECIES_PRETTY,))
plt.savefig('distribution%s.jpg'% (suffix,), dpi=400)

# Comparison of rankings by MGM vs init model pred - 2D
# Read in rankings
rankings = pd.read_csv(rankings_path)
# Compute number of sequences that have a cost, i.e. are "ranked"
n_ranked = len(rankings.loc[rankings['Cost'] != 'undefined'])
# Sort by cost, putting undefined at the bottom
rankings['Cost'] = pd.to_numeric(rankings['Cost'], errors='coerce')
rankings.sort_values(by=['Cost'], inplace=True)
rankings['Cost'] = rankings['Cost'].fillna("undefined")
# Assign rank column as sorted by cost
rankings['MGM_rank'] = range(1, len(rankings) + 1)
rankings['MGM_rank'].loc[rankings['Cost'] == 'undefined'] = len(rankings) + 1
# Sort by initial pred and assign rank column as sorted by initial pred
rankings = rankings.sort_values(by=['Initial pred'], ascending=False)
rankings['model_rank'] = range(1, len(rankings) + 1)
# Compute and assign rank change
rankings['rank_change'] = rankings['model_rank'] - rankings['MGM_rank']
# Save updated rankings
rankings.to_csv('rankings_corrected_with_ranks%s.csv' % (suffix,))

rankings1 = rankings.loc[rankings['Species'] == WITHHELD_SPECIES]
rankings0 = rankings.loc[rankings['Species'] != WITHHELD_SPECIES]
#rankings_spill_seq = rankings.loc[rankings['defline'] == SPILL_SEQ_DEFLINE]

plt.clf()
plt.scatter(rankings1['model_rank'], rankings1['MGM_rank'], s=50, facecolors='none', edgecolors='r', label=WITHHELD_SPECIES_PRETTY)
plt.scatter(rankings0['model_rank'], rankings0['MGM_rank'], s=50, facecolors='none', edgecolors='b', label='Other groups', alpha=0.7)
#plt.scatter(rankings_spill_seq['model_rank'], rankings_spill_seq['MGM_rank'], s=15, facecolors='black', edgecolors='black', marker="*", label=SPILL_SEQ_PRETTY)
plt.xlabel('Risk ranking by initial model score')
plt.ylabel('Risk ranking by MGM-d')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize='small')
plt.title('Comparison of ranking methods')
plt.xlim(1,n_ranked)
plt.ylim(1,n_ranked)
plt.plot([1, n_ranked], [1, n_ranked], color = 'black', linewidth = 0.5, linestyle='--')
plt.savefig('rank_scatter%s.jpg' % (suffix,), dpi=400, bbox_inches="tight")

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
#plt.scatter(x=spill_seq_profit, y = 1, marker="*", label="RaTG13", linestyle="--", color="black")

plt.xlabel('Relative increase in ranking via MGM')
plt.ylabel('Count')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.title('Ranking change relative to ranking by model score')
plt.savefig('rank_change_distribution%s.jpg' % (suffix,), dpi=400)




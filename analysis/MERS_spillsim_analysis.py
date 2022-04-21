import os
import pickle

import pandas as pd

from mgm.analysis.rankings_analysis import make_ranking_plots
from mgm.analysis.trajectory_analysis import conf_trajectory
from mgm.common.sequence import unaligned_idx_to_mult_align_idx
from mgm.common.utils import set_data_directory
import matplotlib.pyplot as plt
import numpy as np

from mgm.data.kuzmin_data import load_kuzmin_data
from mgm.pipelines.spillover_simulation import analyze_variants, analyze_variants_at_threshold, select_non_withheld_pos

########################################################################################################################
# Set analysis parameters
########################################################################################################################
data_dir = "spillover_simulation_MERS_v2"
WITHHELD_SPECIES = 'Middle_East_respiratory_syndrome_coronavirus'
WITHHELD_SPECIES_PRETTY = 'MERS'
THRESHOLD = 0.04
keep_final_seq = True
LIM = None  # Scatter plot limit

params = {}
params['data_dir'] = data_dir
params['WITHHELD_SPECIES'] = WITHHELD_SPECIES
params['WITHHELD_SPECIES_PRETTY'] = WITHHELD_SPECIES_PRETTY
params['THRESHOLD'] = THRESHOLD
params['keep_final_seq'] = keep_final_seq
params['LIM'] = LIM

if THRESHOLD is None:
    suffix = ''
else:
    assert (THRESHOLD <= 1)
    suffix = '_' + str(int(THRESHOLD * 100))

if keep_final_seq == True:
    suffix = suffix + "_keepfinal"

rankings_path = 'rankings%s.csv' % (suffix,)

params['suffix'] = suffix
params['rankings_path'] = rankings_path
########################################################################################################################

# Set directory to where results are
set_data_directory(params['data_dir'])

# Load variants
with open(r"variants.pkl", "rb") as f:
    variants = pickle.load(f)

# Legacy compatibility: Check if non-withheld positive sequences are available
if not os.path.isfile("non_withheld_pos_seqs.pkl"):
    _, _, _, _, _, _, _, seqs = load_kuzmin_data()
    non_withheld_pos_seqs = select_non_withheld_pos(seqs, species_withheld=params['WITHHELD_SPECIES'])
    with open("non_withheld_pos_seqs.pkl", 'wb') as file:
        pickle.dump(non_withheld_pos_seqs, file, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(r"non_withheld_pos_seqs.pkl", "rb") as f:
        non_withheld_pos_seqs = pickle.load(f)

# Re-analyze at original threshold if analysis code was updated since experiment
if not os.path.isfile("rankings_%s_keepfinal.csv" % (str(int(variants[0].confidence_threshold * 100)),)):
    analyze_variants(variants, non_withheld_pos_seqs, filename= "rankings_%s_keepfinal.csv" % (str(int(variants[0].confidence_threshold * 100)),))

# Analyze at new threshold
if THRESHOLD is not None and not os.path.isfile(rankings_path):
    analyze_variants_at_threshold(variants, non_withheld_pos_seqs, THRESHOLD, rankings_path, keep_final_seq=keep_final_seq)

# Make confidence trajectories
conf_trajectory(variants, x_col='change_number', sigma=2)
conf_trajectory(variants, x_col='cost', sigma=None)

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

make_ranking_plots(baseline='Initial pred', **params)
make_ranking_plots(baseline='Edit Distance to Closest Positive', **params)
make_ranking_plots(baseline='Blossom Similarity to Closest Positive', **params)




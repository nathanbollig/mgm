import pickle

import pandas as pd

from mgm.analysis.rankings_analysis import make_ranking_plots
from mgm.analysis.trajectory_analysis import conf_trajectory
from mgm.common.sequence import unaligned_idx_to_mult_align_idx
from mgm.common.utils import set_data_directory
import matplotlib.pyplot as plt
import numpy as np
from mgm.pipelines.spillover_simulation import analyze_variants, reanalyze_variants

########################################################################################################################
# Set analysis parameters
########################################################################################################################
data_dir = "spillover_simulation_SARS2_v2"
SPILL_SEQ_DEFLINE = 'RaTG13|QHR63300|Bat|SARS_CoV_2'
SPILL_SEQ_PRETTY = 'RaTG13'
WITHHELD_SPECIES = 'SARS_CoV_2'
WITHHELD_SPECIES_PRETTY = 'SARS CoV 2'
THRESHOLD = 0.99
keep_final_seq = True
LIM = None  # Scatter plot limit

params = {}
params['data_dir'] = data_dir
params['SPILL_SEQ_DEFLINE'] = SPILL_SEQ_DEFLINE
params['SPILL_SEQ_PRETTY'] = SPILL_SEQ_PRETTY
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

rankings_path = 'rankings%s.csv' % (suffix,)

if keep_final_seq == True:
    suffix = suffix + "_keepfinal"

params['suffix'] = suffix
params['rankings_path'] = rankings_path
########################################################################################################################

# Set directory to where results are
set_data_directory(data_dir)

# Load variants
with open(r"variants.pkl", "rb") as f:
    variants = pickle.load(f)

# Make confidence trajectories
conf_trajectory(variants, x_col='change_number', sigma=2)
conf_trajectory(variants, x_col='cost', sigma=None)

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
plt.savefig('distribution%s.jpg' % (suffix,), dpi=400)

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

if WITHHELD_SPECIES == 'SARS_CoV_2':
    SARSCoV2_endpoints(variants)

make_ranking_plots(baseline='Initial pred', **params)
make_ranking_plots(baseline='Edit Distance to Closest Positive', **params)
make_ranking_plots(baseline='Blossom Similarity to Closest Positive', **params)




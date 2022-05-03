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
# data_dir = "spillover_simulation_SARS2_v2"
# SPILL_SEQ_DEFLINE = 'RaTG13|QHR63300|Bat|SARS_CoV_2'
# SPILL_SEQ_PRETTY = 'RaTG13'
# WITHHELD_SPECIES = 'SARS_CoV_2'
# WITHHELD_SPECIES_PRETTY = 'SARS CoV 2'
# THRESHOLD = 0.99
# keep_final_seq = False
# LIM = None  # Scatter plot limit
#
# params = {}
# params['data_dir'] = data_dir
# params['SPILL_SEQ_DEFLINE'] = SPILL_SEQ_DEFLINE
# params['SPILL_SEQ_PRETTY'] = SPILL_SEQ_PRETTY
# params['WITHHELD_SPECIES'] = WITHHELD_SPECIES
# params['WITHHELD_SPECIES_PRETTY'] = WITHHELD_SPECIES_PRETTY
# params['THRESHOLD'] = THRESHOLD
# params['keep_final_seq'] = keep_final_seq
# params['LIM'] = LIM
########################################################################################################################

def spillsim_analysis_pipeline(**params):
    # Set directory to where results are
    set_data_directory(params['data_dir'])

    # Load variants
    with open(r"variants.pkl", "rb") as f:
        variants = pickle.load(f)

    # Define suffix and ranking file path
    if params['THRESHOLD'] is None:
        suffix = "_%s_keepfinal" % (str(int(variants[0].confidence_threshold * 100)),)
    else:
        assert (params['THRESHOLD'] <= 1)
        suffix = '_' + str(int(params['THRESHOLD'] * 100))

    if params['THRESHOLD'] is not None and params['keep_final_seq'] == True:
        suffix = suffix + "_keepfinal"

    rankings_path = 'rankings%s.csv' % (suffix,)
    params['rankings_path'] = rankings_path
    params['suffix'] = suffix

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

    # Make confidence trajectories
    conf_trajectory(variants, x_col='change_number', sigma=2)
    conf_trajectory(variants, x_col='cost', sigma=None)

    # Analyze at new threshold
    if params['THRESHOLD'] is not None and not os.path.isfile(rankings_path):
        analyze_variants_at_threshold(variants, non_withheld_pos_seqs, params['THRESHOLD'], rankings_path, keep_final_seq=params['keep_final_seq'])

    # Where are SARS CoV 2 mutations made?
    def get_positions(variants, confidence_threshold=0.95):
        positions = []
        spill_seq_positions = set()
        for variant in variants:
            position_set = set()  # Keep set of positions within each variant, i.e. only count 1 mutation per position per sequence at indicated conf threshold
            if variant.init_seq.species == params['WITHHELD_SPECIES']:
                for i, sub_dict in enumerate(variant.substitution_data):
                    if sub_dict['pred_proba'] <= confidence_threshold:
                        index = sub_dict['pos_to_change']
                        if variant.init_seq.defline == params.get('SPILL_SEQ_DEFLINE'):
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
        plt.scatter(x=x_center, y=3, marker="*", linestyle="--", color="black", label="%s at conf=.75" % (params.get('SPILL_SEQ_PRETTY'),) if i==0 else "")

    plt.legend(fontsize='small')
    plt.title('Mutations suggested in %s' % (params['WITHHELD_SPECIES_PRETTY'],))
    plt.savefig('distribution%s.jpg' % (suffix,), dpi=400)

    # Finding regions for SARS CoV 2
    def SARSCoV2_endpoints(variants):
        # First get representative variant
        for i,variant in enumerate(variants):
            if variant.init_seq.defline == params.get('SPILL_SEQ_DEFLINE'):
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

    if params['WITHHELD_SPECIES'] == 'SARS_CoV_2':
        SARSCoV2_endpoints(variants)

    # Create ranking plots
    make_ranking_plots(baseline='Initial pred', **params)
    make_ranking_plots(baseline='Edit Distance to Closest Positive', **params)
    make_ranking_plots(baseline='Blossom Similarity to Closest Positive', **params)




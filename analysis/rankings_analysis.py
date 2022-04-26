"""
Functions for analyzing and visualizing rankings from spillsim experiments.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def add_rank_values(rankings, baseline, suffix):
    # Compute number of sequences that have a cost, i.e. are "ranked"
    n_ranked = len(rankings.loc[rankings['Cost'] != 'undefined'])
    # Sort by cost, putting undefined at the bottom
    rankings['Cost'] = pd.to_numeric(rankings['Cost'], errors='coerce')
    rankings.sort_values(by=['Cost', 'Final Pred'], ascending=[True, False], inplace=True)
    rankings['Cost'] = rankings['Cost'].fillna("undefined")
    # Assign rank column as sorted by cost then final pred
    rankings['MGM_rank'] = range(1, len(rankings) + 1)
    #rankings['MGM_rank'].loc[rankings['Cost'] == 'undefined'] = len(rankings) + 1  # Assigns all undef cost to last rank
    # Sort by baseline and assign rank column as sorted by baseline
    if baseline == 'Initial pred':
        ascending = False
    elif baseline == 'Edit Distance to Closest Positive':
        ascending = True
    elif baseline == 'Blossom Similarity to Closest Positive':
        ascending = False
    else:
        raise ValueError("Invalid baseline param")
    rankings = rankings.sort_values(by=[baseline], ascending=ascending)
    rankings['%s_rank' % (baseline.lower().replace(' ','_'),)] = range(1, len(rankings) + 1)
    # Save updated rankings
    rankings.to_csv('rankings_corrected_with_ranks%s.csv' % (suffix,))
    return rankings, n_ranked

def make_ranking_plots(rankings_path, WITHHELD_SPECIES, WITHHELD_SPECIES_PRETTY, baseline='Initial pred', LIM=None, suffix='', SPILL_SEQ_DEFLINE=None, SPILL_SEQ_PRETTY=None, **kwargs):
    """
    Make 2D scatter plots and 1D plots comparing MGM rank with ranking according to baseline.

    Input:
        spillsim params
        baseline - 'Initial pred', 'Edit Distance to Closest Positive', 'Blossom Similarity to Closest Positive'
    """
    # Read in rankings
    rankings = pd.read_csv(rankings_path)

    # Add rank values
    rankings, n_ranked = add_rank_values(rankings, baseline, suffix)

    # Get baseline column name
    baseline_rank_col = '%s_rank' % (baseline.lower().replace(' ', '_'),)

    # Compute and assign rank change
    rankings['rank_change'] = rankings[baseline_rank_col] - rankings['MGM_rank']

    # Select rows
    rankings1 = rankings.loc[rankings['Species'] == WITHHELD_SPECIES]
    rankings0 = rankings.loc[rankings['Species'] != WITHHELD_SPECIES]

    if SPILL_SEQ_DEFLINE is not None:
        rankings_spill_seq = rankings.loc[rankings['defline'] == SPILL_SEQ_DEFLINE]

    plt.clf()
    plt.scatter(rankings1[baseline_rank_col], rankings1['MGM_rank'], s=50, facecolors='none', edgecolors='r', label=WITHHELD_SPECIES_PRETTY)
    plt.scatter(rankings0[baseline_rank_col], rankings0['MGM_rank'], s=50, facecolors='none', edgecolors='b', label='Negatives')
    if SPILL_SEQ_PRETTY is not None:
        plt.scatter(rankings_spill_seq[baseline_rank_col], rankings_spill_seq['MGM_rank'], s=15, facecolors='black', edgecolors='black', marker="*", label=SPILL_SEQ_PRETTY)
    plt.xlabel('Risk ranking by %s' % (baseline,))
    plt.ylabel('Risk ranking by MGM-d')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize='small')
    plt.title('Comparison of ranking by MGM to baseline')

    if LIM is None:
        LIM = len(rankings)+1
    if LIM == 'max':
        LIM = len(rankings) + 1

    plt.xlim(1,LIM)
    plt.ylim(1,LIM)
    plt.plot([1, LIM], [1, LIM], color = 'black', linewidth = 0.5, linestyle='--')
    plt.savefig('rank_scatter%s_%s.jpg' % (suffix,baseline_rank_col), dpi=400, bbox_inches="tight")

    # As above, ranking change in 1D
    plt.clf()
    change1 = rankings1['rank_change'].to_list()
    change0 = rankings0['rank_change'].to_list()
    if SPILL_SEQ_DEFLINE is not None:
        spill_seq_profit = rankings_spill_seq['rank_change'].to_list()
    else:
        spill_seq_profit = []
    lower_limit = min(change0 + change1 + spill_seq_profit)
    upper_limit = max(change0 + change1 + spill_seq_profit)
    bins = np.linspace(lower_limit, upper_limit, 50)

    plt.axvline(x=0, linestyle="--", color="black")
    plt.hist(change0, bins, density=False, facecolor='b', edgecolor='k', alpha=0.8, label='Other groups')
    plt.hist(change1, bins, density=False, facecolor='r', edgecolor='k', alpha=0.8, label=WITHHELD_SPECIES_PRETTY)
    if SPILL_SEQ_DEFLINE is not None:
        plt.scatter(x=spill_seq_profit, y = 1, marker="*", label="RaTG13", linestyle="--", color="black")

    plt.xlabel('Relative increase in ranking via MGM')
    plt.ylabel('Count')
    handles, labels = plt.gca().get_legend_handles_labels()
    if SPILL_SEQ_DEFLINE is not None:
        order = [1,0,2]
    else:
        order = [1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.title('Ranking change relative to ranking by baseline (%s)' % (baseline,))
    plt.savefig('rank_change_distribution%s_%s.jpg' % (suffix,baseline_rank_col), dpi=400)
import pickle
from mgm.common.utils import set_data_directory
import matplotlib.pyplot as plt

# Set directory to where results are
set_data_directory("spillover_simulation9")

# Load variants
with open(r"variants.pkl", "rb") as f:
    variants = pickle.load(f)

# Correction for spillover_simulation9
for variant in variants:
    final_pred = variant.substitution_data[-1]['conf']
    variant.confidence_threshold = 0.95
    variant.substitution_data = truncate_mutation_trajectory(variant.substitution_data, variant.confidence_threshold)
    variant.compute_cost("num_differences")

def truncate_mutation_trajectory(substitution_data, confidence_threshold):
    for i, sub_dict in enumerate(substitution_data):
        if sub_dict['pred_proba'] > confidence_threshold:
            substitution_data_truncated = substitution_data[:i+1]
            return substitution_data_truncated
    return substitution_data

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
    for variant in variants:
        if variant.init_seq.species == 'SARS_CoV_2':
            for i, sub_dict in enumerate(variant.substitution_data):
                if sub_dict['pred_proba'] <= confidence_threshold:
                    index = sub_dict['pos_to_change']
                    positions.append(index)
    return positions

positions95 = get_positions(variants, confidence_threshold=0.95)
positions90 = get_positions(variants, confidence_threshold=0.90)
positions85 = get_positions(variants, confidence_threshold=0.85)
positions80 = get_positions(variants, confidence_threshold=0.80)
positions75 = get_positions(variants, confidence_threshold=0.75)

plt.clf()
bins = np.linspace(0, 2396, 100)
plt.hist(positions95, bins, density=False, facecolor='g', edgecolor='k', alpha=0.4, label='conf=.95')
plt.hist(positions85, bins, density=False, facecolor='b', edgecolor='k', alpha=0.4, label='conf=.85')
plt.hist(positions75, bins, density=False, facecolor='r', edgecolor='k', alpha=0.4, label='conf=.75')
plt.xlabel('Position')
plt.ylabel('Count')
plt.legend(fontsize='small')
plt.title('Mutations suggested in SARS CoV 2')
plt.savefig('distribution.jpg', dpi=400)

# Comparison of rankings by MGM vs init model pred - SCATTER
rankings = pd.read_csv('rankings_corrected.csv')
rankings = rankings.loc[rankings['Cost'] != 'undefined']
rankings['MGM_rank'] = range(1, len(rankings) + 1)
rankings = rankings.sort_values(by=['Initial pred'], ascending=False)
rankings['model_rank'] = range(1, len(rankings) + 1)
rankings['rank_change'] = rankings['model_rank'] - rankings['MGM_rank']
rankings.to_csv('rankings_corrected_with_ranks.csv')

rankings1 = rankings.loc[rankings['Species'] == 'SARS_CoV_2']
rankings0 = rankings.loc[rankings['Species'] != 'SARS_CoV_2']
rankings_spill_seq = rankings.loc[rankings['defline'] == 'RaTG13|QHR63300|Bat|SARS_CoV_2']

plt.clf()
plt.scatter(rankings1['model_rank'], rankings1['MGM_rank'], s=50, facecolors='none', edgecolors='r', label='SARS CoV 2')
plt.scatter(rankings0['model_rank'], rankings0['MGM_rank'], s=50, facecolors='none', edgecolors='b', label='Other groups')
plt.scatter(rankings_spill_seq['model_rank'], rankings_spill_seq['MGM_rank'], s=15, facecolors='black', edgecolors='black', marker="*", label='RaTG13')
plt.xlabel('Risk ranking by initial model score')
plt.ylabel('Risk ranking by MGM-d')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize='small')
plt.title('Comparison of ranking methods')
m = len(rankings)+1
plt.xlim(1,m)
plt.ylim(1,m)
plt.plot([1, m], [1, m], color = 'black', linewidth = 0.5, linestyle='--')
plt.savefig('rank_scatter.jpg', dpi=400, bbox_inches="tight")

# As above, ranking profit
plt.clf()
change1 = rankings1['rank_change'].to_list()
change0 = rankings0['rank_change'].to_list()
spill_seq_profit = rankings_spill_seq['rank_change'].to_list()
lower_limit = min(change0 + change1 + spill_seq_profit)
upper_limit = max(change0 + change1 + spill_seq_profit)
bins = np.linspace(lower_limit, upper_limit, 50)

plt.axvline(x=0, linestyle="--", color="black")
plt.hist(change0, bins, density=False, facecolor='b', edgecolor='k', alpha=0.8, label='Other groups')
plt.hist(change1, bins, density=False, facecolor='r', edgecolor='k', alpha=0.8, label='SARS CoV 2')
plt.scatter(x=spill_seq_profit, y = 1, marker="*", label="RaTG13", linestyle="--", color="black")

plt.xlabel('Relative increase in ranking via MGM')
plt.ylabel('Count')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.title('Ranking change relative to ranking by model score')
plt.savefig('rank_change_distribution.jpg', dpi=400)




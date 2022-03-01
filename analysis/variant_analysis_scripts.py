import pickle
from mgm.common.utils import set_data_directory
import matplotlib.pyplot as plt

# Set directory to where results are
set_data_directory("spillover_simulation8")

# Load variants
with open(r"variants.pkl", "rb") as f:
    variants = pickle.load(f)



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
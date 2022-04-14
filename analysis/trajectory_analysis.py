"""
Functions for analyzing and visualizing mutation trajectories stored in Variant objects.
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def conf_vs_change_number(variants, sigma=2):
    """
    Draws a plot of confidence vs. change number for a set of variants. Positive labels are red, and negatives are blue.
    If sigma is not None, uses 1D Gaussian filter to smooth the confidence function.

    Inputs:
        variants - list of Variant objects
        sigma - if None, will plot cost directly; otherwise uses 1D Gaussian filtering
    """
    plt.clf()
    for i, variant in enumerate(variants):
        # Retrieve data from variant
        y = variant.init_seq.y
        init_conf = variant.init_pred
        data = pd.DataFrame(variant.substitution_data)
        # Assemble plot data
        change_num = [0]
        conf = [init_conf]
        if len(data) > 0:  # Check that there was a trajectory, otherwise plot initial point
            change_num.extend(data.change_number.tolist())
            conf.extend(data.conf.tolist())
            assert(len(change_num)==len(conf))
            # Gaussian filter to smooth cost
            if sigma is not None:
                conf = gaussian_filter1d(conf, sigma=sigma)
        # Add to plot
        if y == 1:
            color = 'red'
            label = 'Positive'
        else:
            color = 'blue'
            label = 'Negative'
        plt.plot(change_num, conf, color=color, label=label, linewidth=0.8, alpha=0.6)

    plt.title("Confidence trajectories throughout mutation")
    plt.xlabel("Change number (iteration)")
    plt.ylabel("Confidence")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    plt.grid()
    plt.savefig("conf_vs_change_number.jpg", dpi=400, bbox_inches="tight")

def conf_vs_cost(variants, sigma=2):
    """
    Draws a plot of confidence vs. cost for a set of variants. Positive labels are red, and negatives are blue.
    If sigma is not None, uses 1D Gaussian filter to smooth the confidence function.

    Inputs:
        variants - list of Variant objects
        sigma - if None, will plot cost directly; otherwise uses 1D Gaussian filtering
    """
    plt.clf()
    for i, variant in enumerate(variants):
        # Retrieve data from variant
        y = variant.init_seq.y
        init_conf = variant.init_pred
        init_cost = 0
        data = pd.DataFrame(variant.substitution_data)
        # Assemble plot data
        cost = [init_cost]
        conf = [init_conf]
        if len(data) > 0:  # Check that there was a trajectory, otherwise plot initial point
            cost.extend(data.cost.tolist())
            conf.extend(data.conf.tolist())
            assert(len(cost)==len(conf))
            # Gaussian filter to smooth cost
            if sigma is not None:
                conf = gaussian_filter1d(conf, sigma=sigma)
        # Add to plot
        if y == 1:
            color = 'red'
            label = 'Positive'
        else:
            color = 'blue'
            label = 'Negative'
        plt.plot(cost, conf, color=color, label=label, linewidth=0.8, alpha=0.6)

    plt.title("Confidence trajectories throughout mutation")
    plt.xlabel("Cost")
    plt.ylabel("Confidence")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    plt.grid()
    plt.savefig("conf_trajectories.jpg", dpi=400, bbox_inches="tight")
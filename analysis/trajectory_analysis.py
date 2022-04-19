"""
Functions for analyzing and visualizing mutation trajectories stored in Variant objects.
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def conf_trajectory(variants, x_col='change_number', sigma=2):
    """
    Draws a plot of confidence vs. change number or cost for a set of variants. Positive labels are red, and negatives are blue.
    If sigma is not None, uses 1D Gaussian filter to smooth the confidence function.

    Inputs:
        variants - list of Variant objects
        x_col - indicates variable to be plotted on x axis ('change_number' or 'cost')
        sigma - if None, will plot cost directly; otherwise uses 1D Gaussian filtering
    """
    plt.clf()
    if x_col != 'change_number' and x_col != 'cost':
        raise ValueError('x_col is invalid')

    for i, variant in enumerate(variants):
        # Retrieve data from variant
        y = variant.init_seq.y
        init_conf = variant.init_pred
        data = pd.DataFrame(variant.substitution_data)
        # Assemble plot data
        x_vals = [0]
        conf = [init_conf]
        if len(data) > 0:  # Check that there was a trajectory, otherwise plot initial point
            x_vals.extend(data[x_col].tolist())
            conf.extend(data.conf.tolist())
            assert(len(x_vals)==len(conf))
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
        plt.plot(x_vals, conf, color=color, label=label, linewidth=0.8, alpha=0.6)

    plt.title("Confidence trajectories throughout mutation")
    plt.xlabel(x_col)
    plt.ylabel("Confidence")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    plt.grid()
    plt.savefig("conf_vs_%s%s.jpg" % (x_col, '_sig%s' % (sigma,) if sigma is not None else ''), dpi=400, bbox_inches="tight")

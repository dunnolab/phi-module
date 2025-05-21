import wandb
import pandas as pd

import os
import random
import numpy as np
from typing import List, Dict, Union

import seaborn as sns
import matplotlib.pyplot as plt


custom_palette = ["#2f3677ff", "#717171ff", "#9fa4d4ff", "#4049a3ff", "#676fbcff", "#4b4b4bff"]


sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "font.family": "sans-serif",  # change to 'sans-serif' for modern look
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "legend.frameon": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def _cdf_with_replacement(i,n,N):
    return (i/N)**n


def _compute_variance(N, cur_data, expected_max_cond_n, pdfs):
    """
    this computes the standard error of the max.
    this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
    is stated in the last sentence of the summary on page 10 of 
    http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
    """
    variance_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_var = 0
        for i in range(N):
            cur_var += (cur_data[i] - expected_max_cond_n[n])**2 * pdfs[n][i]
        cur_var = np.sqrt(cur_var)
        variance_of_max_cond_n.append(cur_var)
    return variance_of_max_cond_n
    

# this implementation assumes sampling with replacement for computing the empirical cdf
def samplemax(validation_performance):
    validation_performance = list(validation_performance)
    validation_performance.sort()
    N = len(validation_performance)
    pdfs = []
    for n in range(1,N+1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1,N+1):
            F_Y_of_y.append(_cdf_with_replacement(i,n,N))


        f_Y_of_y = []
        cur_cdf_val = 0
        for i in range(len(F_Y_of_y)):
            f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
            cur_cdf_val = F_Y_of_y[i]
        
        pdfs.append(f_Y_of_y)

    expected_max_cond_n = []
    for n in range(N):
        # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
        cur_expected = 0
        for i in range(N):
            cur_expected += validation_performance[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)


    var_of_max_cond_n = _compute_variance(N, validation_performance, expected_max_cond_n, pdfs)

    return {"mean":expected_max_cond_n, "var":var_of_max_cond_n, "max": np.max(validation_performance),
            "min":np.min(validation_performance)}


def one_plot(data, data_name, logx=False, plot_errorbar=True, avg_time=0, performance_metric="accuracy", baseline_error=None):
    linestyle = "-"
    linewidth = 2.5
    errorbar_alpha = 0.2
    fontsize = 16
    x_axis_time = avg_time != 0

    fig, cur_ax = plt.subplots(figsize=(7, 5))
    cur_ax.set_ylabel(f"Expected validation MAE (eV)", fontsize=fontsize)
    cur_ax.set_xlabel("Training duration" if x_axis_time else "Hyperparameter assignments", fontsize=fontsize)

    if logx:
        cur_ax.set_xscale('log')

    means = np.array(data['mean'])
    vars = np.array(data['var'])
    max_acc = data['max']
    min_acc = data['min']

    x_axis = np.arange(1, len(means) + 1)
    if x_axis_time:
        x_axis = avg_time * x_axis

    if plot_errorbar:
        lower = np.clip(means - vars, min_acc, None)
        upper = np.clip(means + vars, None, max_acc)
        cur_ax.fill_between(x_axis, lower, upper, alpha=errorbar_alpha, color=custom_palette[1])

    means_flipped = np.max(means) - means + np.min(means)

    cur_ax.plot(x_axis, means_flipped, linestyle=linestyle, linewidth=linewidth,
                color=custom_palette[2], label=r'$\boldsymbol{\Phi}$-SchNet')

    cur_ax.axhline(
        y=baseline_error, # input the baseline score here
        linestyle='--',
        color=custom_palette[5],
        linewidth=3,
        label=r'SchNet'
    )

    cur_ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    cur_ax.set_xlim([x_axis[0], x_axis[-1]])
    cur_ax.locator_params(axis='y', nbins=6)

    cur_ax.legend(fontsize=fontsize - 2, loc='best')
    plt.tight_layout()
    plt.savefig("plots/evp_schnet.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def get_sweep_data(sweep_id):
    api = wandb.Api()

    entity = "your-entity"
    project = "phi-module"

    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs

    data = []
    for run in runs:
        if run.state != "finished":
            continue 
        config = run.config
        summary = run.summary
        row = {**config, "test_loss": summary.get("test_loss", None)}  
        data.append(row)

    df = pd.DataFrame(data)

    return df['test_loss'].values


if __name__ == '__main__':
    # Input here your WandB sweep ID
    # Also find the baseline error of a model
    sweep_id = None 
    baseline_error = None

    test_losses_phi_module = get_sweep_data(sweep_id)
    test_losses_phi_module = [x for x in test_losses_phi_module if not np.isnan(x)]
    data_phi_module = samplemax(test_losses_phi_module)

    one_plot(data_phi_module, "Experiment Name", logx=False, plot_errorbar=False, avg_time=0, baseline_error=baseline_error)



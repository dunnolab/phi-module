import matplotlib.pyplot as plt
import numpy as np

# Model labels and data
models = ['SchNet', 'DimeNet++', 'PaiNN', 'GemNet-T', r'$\text{E}_2\text{GNN}$']
data_fracs = [5, 25, 50]
custom_palette = ["#2f3677ff", "#717171ff", "#9fa4d4ff", "#4049a3ff", "#676fbcff", "#4b4b4bff"]

# Data retrived from distinct runs
baseline_data = [
    [400.1, 162.9, 126.4],
    [205.2, 114.5, 90.4],
    [263.8, 127.3, 99.0],
    [243.3, 113.8, 85.4],
    [252.9, 126.0, 93.9],
]

enhanced_data = [
    [343.5, 151.2, 117.5],
    [206.1, 103.4, 80.2],
    [258.1, 126.3, 89.7],
    [239.5, 101.6, 75.6],
    [270.0, 125.5, 93.0],
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)

for i, model in enumerate(models):
    ax = axes[i]
    
    ax.plot(data_fracs, baseline_data[i], linestyle='--', marker='o', color=custom_palette[4],
            label='Baseline', linewidth=4)
    ax.plot(data_fracs, enhanced_data[i], linestyle='-', marker='s', color=custom_palette[0],
            label=r'$\boldsymbol{\Phi}\mathbf{-Module}$', linewidth=4)

    ax.set_title(model, fontsize=17)
    ax.set_xlabel("Data (%)", fontsize=17)
    ax.set_xticks(data_fracs)
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.tick_params(axis='both', which='major', labelsize=15)

    if i == 0:
        ax.set_ylabel("OE62 Test MAE (meV)", fontsize=17)

axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=15)

plt.tight_layout()
plt.savefig("plots/data_efficiency.pdf", dpi=300)
plt.show()

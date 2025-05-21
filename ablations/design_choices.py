import matplotlib.pyplot as plt
import numpy as np

custom_palette = ["#2f3677ff", "#717171ff", "#9fa4d4ff", "#4049a3ff", "#676fbcff", "#4b4b4bff"]

ablation_markers = ['o', 's', 'D', '^', 'H']

ablation_labels = [
    r'Baseline',
    r'Random $\mathbf{L}$',
    r'No $L_{\mathrm{PDE}}$',
    r'Random $\mathbf{L}$ + No $L_{\mathrm{PDE}}$',
    r'$\boldsymbol{\Phi}\mathbf{-Module}$'
]

method_names = ["SchNet", "PaiNN", "DimeNet++", "GemNet-T", r'$E_2$GNN']

data = np.array([
    [95.5, 92.8, 91.1, 93.9, 84.0],  # SchNet
    [76.0, 73.4, 73.4, 73.7, 66.0],  # PaiNN
    [67.1, 63.2, 61.2, 63.0, 58.5],  # DimeNet++
    [63.5, 58.8, 57.1, 56.5, 56.1],  # GemNet-T
    [68.5, 67.2, 65.8, 68.5, 65.2],  # E2GNN
])

fig, ax = plt.subplots(figsize=(8, 5))
x_jitter = np.linspace(-0.06, 0.06, len(ablation_markers))

for method_idx, (method_data, color) in enumerate(zip(data, custom_palette)):
    for ablation_idx, value in enumerate(method_data):
        x = method_idx + x_jitter[ablation_idx]
        ax.scatter(
            x, value,
            color=color,
            edgecolor='k',
            linewidth=0.8,
            s=130,
            marker=ablation_markers[ablation_idx],
            label=ablation_labels[ablation_idx] if method_idx == 0 else None,
            alpha=0.95
        )

ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=15, ha='right', fontsize=17)
ax.set_ylabel("OE62 Test MAE (meV)", fontsize=17)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(title="Ablation Type", frameon=False, fontsize=15, title_fontsize=12, loc="upper right")

plt.tight_layout()
plt.savefig("plots/design_choices.pdf", dpi=300)
plt.show()

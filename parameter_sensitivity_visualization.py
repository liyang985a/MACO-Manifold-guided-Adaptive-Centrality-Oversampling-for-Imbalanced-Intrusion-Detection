import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# =========================================================
# 1. Global Font and Style Configuration
# =========================================================
# Increase font sizes globally to prevent visual crowding
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.unicode_minus': False,
    'font.size': 14,  # Global default font size
    'axes.titlesize': 16,  # Subplot title font size
    'axes.labelsize': 14,  # Axis label font size
    'xtick.labelsize': 13,  # X-axis tick font size
    'ytick.labelsize': 13,  # Y-axis tick font size
    'legend.fontsize': 13  # Legend font size
})

# =========================================================
# 2. Data Definition
# =========================================================
# Global Baseline values
baseline = {
    'Pre': 0.6127,
    'F1': 0.6025,
    'MCC': 0.8651,
    'G_means': 0.6310
}

# Experimental results for each parameter
experiments = {
    'dim_ratio': {
        'x': [0.1, 0.3, 0.5, 0.7],
        'Pre': [0.5935, 0.6210, 0.5956, 0.5638],
        'F1': [0.5944, 0.5955, 0.5943, 0.5792],
        'MCC': [0.8609, 0.8688, 0.8675, 0.8576],
        'G_means': [0.6312, 0.6324, 0.6191, 0.6025],
    },
    'k_sim': {
        'x': [1, 5, 7],
        'Pre': [0.6304, 0.6111, 0.5494],
        'F1': [0.5916, 0.5901, 0.5676],
        'MCC': [0.8575, 0.8601, 0.8226],
        'G_means': [0.6397, 0.6305, 0.6164],
    },
    'k_opp': {
        'x': [1, 5, 7],
        'Pre': [0.5924, 0.6160, 0.5382],
        'F1': [0.5836, 0.5964, 0.5621],
        'MCC': [0.8586, 0.8656, 0.8400],
        'G_means': [0.6160, 0.6341, 0.6106],
    },
    'top_n_opp_classes': {
        'x': [1, 2, 5],
        'Pre': [0.5223, 0.6000, 0.6034],
        'F1': [0.5562, 0.5748, 0.5844],
        'MCC': [0.8293, 0.8572, 0.8503],
        'G_means': [0.6145, 0.6196, 0.6365],
    },
    'K': {
        'x': [1, 3, 7],
        'Pre': [0.5278, 0.6314, 0.5321],
        'F1': [0.5484, 0.5763, 0.5617],
        'MCC': [0.8379, 0.8674, 0.8268],
        'G_means': [0.5939, 0.6242, 0.6107],
    },
    'n_generate_per_sample': {
        'x': [1, 3, 5],
        'Pre': [0.6010, 0.5040, 0.5023],
        'F1': [0.5820, 0.5451, 0.5225],
        'MCC': [0.8612, 0.8307, 0.7889],
        'G_means': [0.6240, 0.6054, 0.5945],
    }
}

# =========================================================
# 3. Plotting Initialization
# =========================================================
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

metrics = ['Pre', 'F1', 'MCC', 'G_means']

# Use a professional, high-contrast color palette (Seaborn Deep style)
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# =========================================================
# 4. Rendering Subplots
# =========================================================
for idx, (param, data) in enumerate(experiments.items()):
    ax = axes[idx]
    x = data['x']
    for i, metric in enumerate(metrics):
        # Emphasize experimental lines and markers (first visual hierarchy)
        ax.plot(x, data[metric], color=colors[i], marker='o', markersize=8, linewidth=2.5, zorder=3)
        # De-emphasize Baseline dashed lines to reduce crowding (alpha=0.4)
        ax.axhline(baseline[metric], linestyle='--', color=colors[i], alpha=0.4, linewidth=2, zorder=2)

    # Subplot formatting
    ax.set_title(f"Parameter: {param}")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0.48, 0.90)

    # Soften grid lines to prevent interference with data visualization
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# =========================================================
# 5. Global Legend Configuration
# =========================================================
# Simplify the legend from 8 items to 6 items to clarify the logic
custom_lines = [
    Line2D([0], [0], color=colors[0], lw=3, label='Pre'),
    Line2D([0], [0], color=colors[1], lw=3, label='F1'),
    Line2D([0], [0], color=colors[2], lw=3, label='MCC'),
    Line2D([0], [0], color=colors[3], lw=3, label='G_means'),
    Line2D([0], [0], color='gray', lw=2.5, marker='o', markersize=8, label='Experiment (Solid)'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Baseline (Dashed)')
]

# Place the unified legend at the top center of the entire figure
fig.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=3, frameon=False)

# Adjust subplot layout to leave enough vertical space for the global legend
plt.tight_layout(rect=[0, 0, 1, 0.92])

# =========================================================
# 6. Save and Display
# =========================================================
print("⏳ Saving high-resolution plots to the current directory...")

# Save as high-resolution PNG (600 DPI is standard for top-tier journals)
plt.savefig("Parameter_Sensitivity_Analysis.png", dpi=600, bbox_inches='tight')

# Save as Vector PDF (Scalable without losing quality, ideal for LaTeX/Word)
plt.savefig("Parameter_Sensitivity_Analysis.pdf", format='pdf', bbox_inches='tight')

print("✅ Plots saved successfully!")

# Display the figure window (Must be called after savefig)
plt.show()
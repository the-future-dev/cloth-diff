#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import CheckButtons

# --- Configuration ---
sns.set_theme(style="whitegrid", font_scale=1.1, rc={"grid.alpha": 0.3})
plt.rcParams['font.family'] = 'sans-serif'

# --- Plot Parameters ---
colors = sns.color_palette("tab10", 6)
plot_params = {
    'linewidth': 2,
    'markersize': 8,
}
fill_alpha = 0.15

# Updated markers: squares for Image and State, circles for the rest
markers = {'State Only': 's', 'Image Only': 's'}
default_marker = 'o'
linestyles = ['-', '--', '-.', ':', '-', '--']

# --- Data ---
dataset_sizes = np.array([40, 200, 1000, 4000, 8000])
data = {
    # Baselines
    "State Only": (
        np.array([0.38741, 0.47251, 0.52178, 0.54924, 0.58481]),
        np.array([0.30463, 0.34743, 0.28954, 0.30059, 0.30447])
    ),
    "Image Only": (
        np.array([-0.12789, 0.03854, 0.32911, 0.47981, 0.54843]),
        np.array([ 0.54948, 0.17191, 0.35630, 0.36152, 0.30119])
    ),
    # Privileged-information recipes
    "NoEnc Concat (dropout=0.37)": (
        np.array([-0.0675, 0.061, 0.1882, 0.333, 0.54037]),
        np.array([0.39924, 0.16782, 0.32291, 0.0, 0.3129])
    ),
    "MLP Enc Concat (dropout=0.37)": (
        np.array([0.01258, -0.0227, 0.2367, 0.557, 0.55116]),
        np.array([0.16278, 0.15442, 0.34073, 0.31905, 0.29718])
    ),
    "MLP Enc Sum (dropout=0.37)": (
        np.array([0.001, 0.004, 0.2577, 0.53621, 0.56275]),
        np.array([0.037, 0.21855, 0.33764, 0.32177, 0.29809])
    ),
    "Transformer Shared Enc, MLP Enc, Concat": (
        np.array([0.002, 0.056, 0.355, 0.54725, 0.55257]),
        np.array([0.0215, 0.16419, 0.39223, 0.34975, 0.28878])
    ),
}

# --- Create output directory ---
output_dir = os.path.join('img', 'priv-diff')
os.makedirs(output_dir, exist_ok=True)

# --- Create interactive plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize lines and fill_betweens
lines = []
fills = []
labels = list(data.keys())
visible = [True] * len(labels)  # All lines visible initially

for idx, (label, (perf, std)) in enumerate(data.items()):
    mask = ~np.isnan(perf)
    x = dataset_sizes[mask]
    y = perf[mask]
    yerr = std[mask]
    color = colors[idx % len(colors)]
    
    # Create line with appropriate marker
    marker = markers.get(label, default_marker)
    line, = ax.plot(
        x, y,
        marker=marker,
        linestyle=linestyles[idx % len(linestyles)],
        linewidth=plot_params['linewidth'],
        markersize=plot_params['markersize'],
        color=color,
        label=label,
        markerfacecolor='white',
        markeredgewidth=1.5
    )
    
    # Create fill between
    fill = ax.fill_between(x, y - yerr, y + yerr, alpha=fill_alpha, color=color)
    
    lines.append(line)
    fills.append(fill)

# Add expert performance line
expert_line = ax.axhline(y=0.714, color='black', linestyle='-', linewidth=1.5, label='Expert Performance')
lines.append(expert_line)
labels.append('Expert Performance')
visible.append(True)

# --- Customize plot ---
ax.set_xscale('log')
ax.set_xlabel('Number of Demonstrations (log scale)')
ax.set_ylabel('Normalized Performance')
ax.set_title('Comparative Performance: Bridging Image to State')
ax.set_xticks(dataset_sizes)
ax.set_xticklabels([str(x) for x in dataset_sizes])
ax.set_ylim(-0.4, 1.0)

ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.minorticks_on()

# Create legend
legend = ax.legend(
    fontsize=10, frameon=True, framealpha=0.9,
    loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0
)

# Add check buttons for toggling visibility
plt.subplots_adjust(left=0.2, right=0.75)
rax = plt.axes([0.80, 0.4, 0.15, 0.3])
check = CheckButtons(rax, labels, visible)

def update(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    fills[index].set_visible(not fills[index].get_visible())
    fig.canvas.draw_idle()

check.on_clicked(update)

# --- Save and show ---
plt.tight_layout(pad=1.0)
output_path = os.path.join(output_dir, 'priv_diff_comparison_interactive.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

# Show interactive plot
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import numpy as np

# Set a professional style for the plot with increased base font size
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 28,
    'axes.labelsize': 32,
    'axes.titlesize': 40,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 29,
    'legend.title_fontsize': 35
})

# Updated data from provided list
data = {
    'Model': [
        'GPT-5', 'Gemini-2.5-Pro', 'GPT-o3', 'Gemini-2.5-Flash', 'GPT-4.1',
        'Nemotron-14B', 'Qwen3-14B thinking', 'GPT-4o',
        'Qwen3-8B thinking', 'Sky-T1-7B', 'Sky-T1-mini',
        'Nemotron-7B', 'Qwen3-4B thinking', 'Qwen3-14B non-thinking',
        'AReaL-boba-2-14B', 'AReaL-boba-2-8B',
        'Qwen3-4B non-thinking', 'Qwen3-8B non-thinking'
    ],
    'Accuracy': [
        74.25, 75.25, 69.50, 62.00, 52.25,
        43.75, 43.25, 36.50,
        38.50, 29.50, 29.75,
        37.25, 36.25, 34.25,
        32.50, 31.00,
        28.75, 28.25
    ],
    'Decoding_Token_Length': [
        1610.5, 638, 2504.5, 402.5, 725.5,
        7821.5, 9111, 482.5,
        17889.5, 421, 7092.5,
        6447.5, 19095, 2633,
        11799.5, 16324.5,
        2499.5, 2648.5
    ],
    'Model_Family': [
        'GPT', 'Gemini', 'GPT', 'Gemini', 'GPT',
        'NVIDIA', 'Qwen', 'GPT',
        'Qwen', 'NovaSky', 'NovaSky',
        'NVIDIA', 'Qwen', 'Qwen',
        'inclusionAI', 'inclusionAI',
        'Qwen', 'Qwen'
    ],
    'Model_Size': [
        'XXL', 'XL', 'XL', 'L', 'L',
        'L', 'L', 'XL',
        'L', 'M', 'S',
        'M', 'S', 'L',
        'L', 'M',
        'S', 'M'
    ],
    # Params_B values below are not updated as the prompt didn't provide; can be adjusted if needed
    'Params_B': [
        100, 60, 40, 14, 9,
        14, 14, 8,
        14, 7, 3.5, 7, 4, 14,
        14, 8, 4, 8
    ]
}

size_mapping = {
    'S': 90 * 4,
    'M': 170 * 5,
    'L': 270 * 5.5,
    'XL': 400 * 6.5,
    'XXL': 530 * 7
}

df_all = pd.DataFrame(data)

# --- Hide the desired models ---
models_to_hide = ['Qwen3-4B thinking', 'Qwen3-4B non-thinking']
df_all = df_all[~df_all['Model'].isin(models_to_hide)].reset_index(drop=True)

family_colors = {
    'GPT': '#5EC9C6',
    'Gemini': '#AB94D6',
    'NVIDIA': '#7CB3DE',
    'NovaSky': '#FFD37A',
    'Qwen': '#FFAD89',
    'inclusionAI': '#BCAAA4'
}

fig, ax = plt.subplots(figsize=(24, 10))

ax.set_xscale('log')

# Scatter all datapoints
for _, row in df_all.iterrows():
    size_val = size_mapping.get(row['Model_Size'], 160 * 4.5)
    x = row['Decoding_Token_Length']
    y = row['Accuracy']
    color = family_colors.get(row['Model_Family'], '#808080')
    ax.scatter(
        x,
        y,
        s=size_val,
        c=[color],
        alpha=0.9,
        edgecolors='black',
        linewidth=0.7,
        zorder=3
    )

# Define per-model annotation offsets for clearer layout (manual for each model)
annotation_offsets = {
    "GPT-5":                  (1.07, 3.4,    'center', 'bottom'),
    "Gemini-2.5-Pro":         (1.095, -2.9,  'center', 'top'),
    "GPT-o3":                 (0.95, 3.2,    'right', 'bottom'),
    "Gemini-2.5-Flash":       (1.06, 0.9,    'left', 'bottom'),
    "GPT-4.1":                (1.07, -2.7,   'left', 'top'),
    "Nemotron-14B":           (0.57, 8.0,    'left', 'top'),     # moved even more left and up
    "Qwen3-14B thinking":     (1.57, 8.0,   'right', 'top'),    # even more topright
    "GPT-4o":                 (1.0, 3.8,     'center', 'top'),   # center, top, higher
    "Qwen3-8B thinking":      (1.0, 4.2,     'center', 'top'),   # center, top, higher
    "Sky-T1-7B":              (0.97, 2.55,   'right', 'bottom'),
    "Sky-T1-mini":            (0.97, -3.3,   'right', 'top'),
    "Nemotron-7B":            (0.80, 4.5,    'left', 'top'),     # lefttop, high
    # These two are hidden:
    # "Qwen3-4B thinking":    (1.14, 2.7,    'left', 'bottom'),
    "Qwen3-14B non-thinking": (0.79, 2.5,    'left', 'bottom'),
    "AReaL-boba-2-14B":       (1.0, 4.4,     'center', 'top'),   # center, top, higher
    "AReaL-boba-2-8B":        (0.95, -3.0,   'right', 'top'),
    # "Qwen3-4B non-thinking": (1.06, -2.6,   'left', 'top'),
    "Qwen3-8B non-thinking":   (1.15, 1.4,    'left', 'bottom'),
}

for _, row in df_all.iterrows():
    x = row['Decoding_Token_Length']
    y = row['Accuracy']
    label = row['Model']
    dx, dy, ha, va = annotation_offsets.get(
        label,
        (1.07, 2.1, 'center', 'bottom')
    )
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x * dx, y + dy),
        textcoords='data',
        ha=ha, va=va,
        fontsize=23,
        bbox=dict(boxstyle='round,pad=0.13', fc='white', ec='gray', alpha=0.80, linewidth=0.5),
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.25, alpha=0.86, shrinkA=7.5, shrinkB=2.5),
        zorder=10
    )

ax.set_xlabel('Decoding Token Length (Log Scale)', fontsize=36, fontweight='bold', labelpad=23)
ax.set_ylabel('Accuracy (%)', fontsize=36, fontweight='bold', labelpad=21)
ax.set_title('Model Performance: Accuracy vs. Decoding Token Length', fontsize=46, fontweight='bold', pad=40)
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)

x_min = df_all['Decoding_Token_Length'].min() * 0.65
x_max = df_all['Decoding_Token_Length'].max() * 1.5
ax.set_xlim(left=x_min, right=x_max)
ax.set_ylim(bottom=df_all['Accuracy'].min() - 5, top=df_all['Accuracy'].max() + 5)

ax.grid(True, which='both', linestyle='--', linewidth=0.58, alpha=0.54)

legend_elements_family = [
    Line2D([0], [0], marker='o', color='w', label=family,
           markerfacecolor=color, markersize=36, markeredgecolor='black')
    for family, color in family_colors.items() if family in df_all['Model_Family'].unique()
]

ax.legend(
    handles=legend_elements_family,
    title='Model Family',
    loc='upper right',
    fontsize=29,
    title_fontsize=35,
    frameon=True,
    ncol=2  # Two columns for the legend
)

plt.tight_layout()
plt.savefig('accuracy_vs_tokens_refined_dense.png', dpi=300)
plt.savefig('accuracy_vs_tokens_refined_dense.pdf')
plt.show()

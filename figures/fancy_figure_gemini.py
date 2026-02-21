import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from scipy.optimize import curve_fit

# 1. Prepare the Data
data = [
    ["GPT-4o", "Commercial", 530.02, 61.00],
    ["Sky-T1-7B", "Open Source", 511.45, 58.00],
    ["GPT-4.1", "Commercial", 977.43, 80.50],
    ["Qwen3-235B-A22B-non_thinking", "Open Source", 963.65, 62.00],
    ["GPT-5", "Commercial", 2016.37, 88.50],
    ["GPT-o3", "Commercial", 2280.02, 83.00],
    ["Qwen3-32B-non_thinking", "Open Source", 1695.11, 58.50],
    ["Qwen3-30B-A3B-non_thinking", "Open Source", 1839.41, 58.00],
    ["Qwen3-14B-non_thinking", "Open Source", 2406.07, 60.00],
    ["Qwen3-8B-non_thinking", "Open Source", 2188.66, 52.50],
    ["Gemini-2.5 Pro", "Commercial", 4011.93, 87.50],
    ["Qwen3-4B-non_thinking", "Open Source", 2247.39, 49.00],
    ["Gemini-2.5 Flash", "Commercial", 3970.64, 85.00],
    ["AceReason-Nemotron-14B", "Open Source", 4028.84, 71.50],
    ["Qwen3-32B-thinking", "Open Source", 5217.71, 73.50],
    ["Qwen3-30B-A3B-thinking", "Open Source", 5399.09, 74.50],
    ["AceReason-Nemotron-7B", "Open Source", 4271.45, 57.50],
    ["AReaL-boba-2-32B", "Open Source", 5710.06, 76.00],
    ["Qwen3-14B-thinking", "Open Source", 5775.52, 73.50],
    ["AReal-boba-2-14B", "Open Source", 7133.56, 74.50],
    ["Qwen3-235B-A22B-thinking", "Open Source", 7254.51, 71.50],
    ["Qwen3-8B-thinking", "Open Source", 7321.98, 70.00],
    ["AReal-boba-2-8B", "Open Source", 8684.22, 67.00],
    ["Qwen3-4B-thinking", "Open Source", 8942.01, 64.00],
    ["Sky-T1-mini", "Open Source", 8651.71, 50.50]
]

df = pd.DataFrame(data, columns=["Model", "Category", "Tokens", "Accuracy"])

# 2. Helper to extract model size for bubble sizing
def get_size(name):
    # Find numbers followed by 'B', default to 10 if not found (e.g. GPT-4o)
    match = re.search(r'(\d+)B', name)
    if match:
        return int(match.group(1))
    # Assign arbitrary sizes for closed source models based on "vibes" or known info
    if "GPT-5" in name: return 1000
    if "GPT-4" in name: return 500
    if "Gemini" in name: return 400
    return 50 

df['Size'] = df['Model'].apply(get_size)
# Normalize size for plotting
df['PlotSize'] = df['Size'] * 3  # Scale factor

# 3. Calculate the Pareto Curve (Logarithmic Fit on Top Performers)
# We filter for models that are roughly "on the frontier" to fit the curve
# This is a heuristic: take the top accuracy models across the token range
frontier_candidates = df[df['Accuracy'] > 60].sort_values('Tokens')
def log_func(x, a, b):
    return a + b * np.log(x)

# Fit curve to the best models (approximating the black line in your reference)
# We manually select a few key points that look like they form the boundary
# or we can just fit to the high accuracy ones.
popt, _ = curve_fit(log_func, frontier_candidates['Tokens'], frontier_candidates['Accuracy'])

# 4. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors
colors = {'Commercial': '#1f77b4', 'Open Source': '#ff7f0e'}

# Plot Scatter Points
for cat, group in df.groupby('Category'):
    ax.scatter(group['Tokens'], group['Accuracy'], 
               s=group['PlotSize'], 
               c=colors[cat], 
               label=cat, 
               alpha=0.9, 
               edgecolor='black', 
               linewidth=0.5)

# Plot the Curve
x_curve = np.linspace(400, 10000, 500)
y_curve = log_func(x_curve, *popt)
# Shift curve up slightly to act as a "ceiling" or adjust manually
ax.plot(x_curve, y_curve + 2, color='black', linewidth=1.5, linestyle='-', zorder=0) 

# 5. Style Adjustments
ax.set_xscale('log')
ax.set_xlim(400, 10500)
ax.set_ylim(40, 95)

# Clean Spines (Top and Right off)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Labels
ax.set_xlabel("Decoding Token Length (Log Scale)", fontsize=12, weight='bold')
ax.set_ylabel("Accuracy (%)", fontsize=12, weight='bold')

# Grid (Vertical only often looks cleaner for x-axis comparisons)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Save as SVG (Vector) - Crucial for Keynote editing
plt.tight_layout()
plt.savefig("pareto_plot.svg", format='svg')
plt.show()
# combined_models_comparison.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch

# ----------------------------
# Data
# ----------------------------
models = ['Nemotron-7B', 'Sky-T1-7B', 'Gemini 2.5 Pro', 'GPT-5']
latency_ms = [3296, 307, 0, 0]  # Blank for Gemini and GPT-5
avg_tokens = [9051, 494, 8530, 2378]

# Calculate speedup and token reduction for the 7B models only
speedup = latency_ms[0] / latency_ms[1] if latency_ms[1] > 0 else 0  # ≈ 10.7×
token_reduction_7b = avg_tokens[0] / avg_tokens[1]  # ≈ 18.3×
token_reduction_gemini_gpt = avg_tokens[2] / avg_tokens[3]  # ≈ 3.6×

# ----------------------------
# Style and Label Colors
# ----------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelsize': 22,   # Default axis label size
    'axes.titlesize': 28,   # Default title size
    'legend.fontsize': 18,  # Default legend font size
})

# Assign pale/bright colors for each model (one color per model)
MODEL_COLORS = ['#A7CDF7', '#FFE1B3', '#A7CDF7', '#FFE1B3']  # Light blue (first two), pale orange (last two)

# Set aspect ratio to x:y = 12:10 ==> width:height = 12:10
fig, ax1 = plt.subplots(figsize=(12, 10), dpi=110)
ax2 = ax1.twinx()

# ----------------------------
# Bars (very visible, extremely sparse hatching: stars and "o"), one model = one color
# ----------------------------
width = 0.38
x = np.arange(len(models))

# Use very sparse "star" and "o" hatching: one every 12+ spaces for very visible, sparse overlay
lat_hatch = '/'   # 1 star every 11 spaces (length-12): extremely sparse
tok_hatch = '*'   # 1 "o" every 11 spaces (length-12): extremely sparse

# Assign bar colors (one color per model: same for latency and token bar of that model)
lat_bar_colors = MODEL_COLORS
tok_bar_colors = MODEL_COLORS

# Draw latency bars with very sparse stars
bars_lat = ax1.bar(
    x - width / 2,
    latency_ms,
    width,
    color=lat_bar_colors,
    edgecolor='black',
    hatch=lat_hatch,
    linewidth=3.1,     # extra-thick edges
    zorder=3,
    label='Latency (ms)',
    alpha=1.0
)

# Make Gemini and GPT-5 latency bars invisible
bars_lat[2].set_visible(False)
bars_lat[3].set_visible(False)

# Draw token bars with very sparse "o" pattern
bars_tok = ax2.bar(
    x + width / 2,
    avg_tokens,
    width,
    color=tok_bar_colors,
    edgecolor='black',
    hatch=tok_hatch,
    linewidth=3.1,
    zorder=3,
    label='Avg. Tokens',
    alpha=1.0
)

# ----------------------------
# Grid, limits, labels - ENHANCED GRID
# ----------------------------

# Hide ax1 y grid; show token count grid (ax2) on top
# We want the token (right) axis grid to show for both axes
# So, turn off ax1 grid, turn on ax2 grid, and force grid above bars for visibility

ax1.yaxis.grid(False)
ax2.yaxis.grid(
    True,
    linestyle='-',
    alpha=0.40,
    zorder=5,  # above bars
    color='#AA8855',
    linewidth=1.65
)

ax1.set_ylim(0, 3700)
ax2.set_ylim(0, 10500)
ax1.set_xlim(-0.6, 3.6)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=20, color='#121212', fontweight='bold')
ax1.set_ylabel('Latency (ms)', fontsize=22, color='black', fontweight='bold')
ax2.set_ylabel('Avg. Tokens', fontsize=22, color='black', fontweight='bold')
ax1.set_title('Performance Comparison: All Models',
              fontsize=28, pad=22, color='#232323', weight='bold', loc='left')

# Make sure grid lines also appear above all patches
for spine in ['left', 'right']:
    ax1.spines['left'].set_zorder(6)
    ax2.spines['right'].set_zorder(6)

# MODIFIED: Changed axis spine and tick colors to black
ax1.spines['left'].set_color('black')
ax1.yaxis.label.set_color('black')
ax1.tick_params(axis='y', colors='black', labelsize=17)
ax2.spines['right'].set_color('black')
ax2.yaxis.label.set_color('black')
ax2.tick_params(axis='y', colors='black', labelsize=17)

# Simplified the legend: just use the new hatches (very sparse)
handles = [
    Patch(facecolor='#dddddd', edgecolor='black', hatch=lat_hatch, label='Latency (ms)', linewidth=2.7),
    Patch(facecolor='#dddddd', edgecolor='black', hatch=tok_hatch, label='Avg. Tokens', linewidth=2.7)
]
ax1.legend(
    handles=handles,
    loc='upper right',  # Changed location for better fit
    frameon=True,
    fancybox=True,
    ncol=2,
    handleheight=2.1,
    columnspacing=1.7,
    fontsize=18
)

# ----------------------------
# Value labels above each bar
# ----------------------------
def add_value_labels(ax, bars, values, fmt, outline_col, bg_col, visible_only=True):
    for i, (b, v) in enumerate(zip(bars, values)):
        if visible_only and not b.get_visible():
            continue
        cx = b.get_x() + b.get_width() / 2
        cy = b.get_height()
        if v > 0:  # Only show labels for non-zero values
            ax.annotate(fmt(v), (cx, cy), xytext=(0, 12), textcoords='offset points',
                        ha='center', va='bottom', fontsize=19, fontweight='bold',
                        color=outline_col,
                        bbox=dict(boxstyle='round,pad=0.32', fc=bg_col,
                                  ec=outline_col, lw=1.9, alpha=0.965),
                        zorder=6)

NEUTRAL_TEXT_COLOR = '#444444'
NEUTRAL_ARROW_COLOR = '#444466'
NEUTRAL_ARROW_LABEL_BBOX = {'boxstyle': 'round,pad=0.38', 'fc': 'white', 'ec': '#444466', 'lw': 2.1}
add_value_labels(ax1, bars_lat, latency_ms, lambda v: f'{v:,} ms', NEUTRAL_TEXT_COLOR, 'white')
add_value_labels(ax2, bars_tok, avg_tokens, lambda v: f'{v:,}', NEUTRAL_TEXT_COLOR, 'white', visible_only=False)

# ----------------------------
# Curved arrows + callouts
# ----------------------------
lat_left_top = bars_lat[0].get_height()
lat_right_top = bars_lat[1].get_height()
tok_left_top = bars_tok[0].get_height()
tok_right_top = bars_tok[1].get_height()
tok_gemini_top = bars_tok[2].get_height()
tok_gpt_top = bars_tok[3].get_height()

latency_offset_below = 0.045 * lat_left_top
tokens_offset_below_7b = 0.052 * tok_left_top
tokens_offset_below_gemini = 0.052 * tok_gemini_top
latency_offset_above = 0.72 * lat_right_top
tokens_offset_above_7b = 1.05 * tok_right_top
tokens_offset_above_gpt = 0.3 * tok_gpt_top

# (A) Latency: "10.7× faster" (ax1 coordinates) - only for 7B models
latency_center_x0 = x[0] - width / 2
latency_center_x1 = x[1] - width / 2
latency_arrow_shift = 0

start1 = (latency_center_x0 + latency_arrow_shift, lat_left_top - latency_offset_below)
end1 = (latency_center_x1 - latency_arrow_shift, lat_right_top + latency_offset_above)

# Draw arrows with very high zorder to ensure they appear on top
arrow1 = FancyArrowPatch(start1, end1, transform=ax1.transData,
                         connectionstyle="arc3,rad=-0.29",
                         arrowstyle='->', mutation_scale=26,
                         linewidth=5.2,
                         color=NEUTRAL_ARROW_COLOR, zorder=100, clip_on=False)
ax2.add_patch(arrow1)
# Force the arrow to be drawn on top by bringing it to front
arrow1.set_zorder(100)
ax2.annotate(f'{speedup:.1f}× faster',
             xy=((start1[0] + end1[0]) / 2 + 0.3, (start1[1] + end1[1]) / 2 + 1000),
             ha='center', va='center', fontsize=22, fontweight='bold', color=NEUTRAL_TEXT_COLOR,
             bbox=NEUTRAL_ARROW_LABEL_BBOX,
             zorder=101)

# (B) Tokens: "18.3× fewer tokens" for 7B models (ax2 coordinates)
start2 = (x[0] + width / 2, tok_left_top - tokens_offset_below_7b)
end2 = (x[1] + width / 2, tok_right_top + tokens_offset_above_7b)

# Draw arrows with very high zorder to ensure they appear on top
arrow2 = FancyArrowPatch(start2, end2, transform=ax2.transData,
                         connectionstyle="arc3,rad=-0.28",
                         arrowstyle='->', mutation_scale=26,
                         linewidth=5.2,
                         color=NEUTRAL_ARROW_COLOR, zorder=100, clip_on=False)
ax2.add_patch(arrow2)
# Force the arrow to be drawn on top by bringing it to front
arrow2.set_zorder(100)
ax2.annotate(f'{token_reduction_7b:.1f}× fewer tokens',
             xy=((start2[0] + end2[0]) / 2 + 0.48, (start2[1] + end2[1]) / 2 + 2),
             ha='center', va='center', fontsize=22, fontweight='bold', color=NEUTRAL_TEXT_COLOR,
             bbox=NEUTRAL_ARROW_LABEL_BBOX,
             zorder=101)

# (C) Tokens: "3.6× fewer tokens" for Gemini vs GPT-5 (ax2 coordinates)
start3 = (x[2] + width / 2, tok_gemini_top - tokens_offset_below_gemini)
end3 = (x[3] + width / 2, tok_gpt_top + tokens_offset_above_gpt)

# Draw arrows with very high zorder to ensure they appear on top
arrow3 = FancyArrowPatch(start3, end3, transform=ax2.transData,
                         connectionstyle="arc3,rad=-0.28",
                         arrowstyle='->', mutation_scale=26,
                         linewidth=5.2,
                         color=NEUTRAL_ARROW_COLOR, zorder=100, clip_on=False)
ax2.add_patch(arrow3)
# Force the arrow to be drawn on top by bringing it to front
arrow3.set_zorder(100)
ax2.annotate(f'{token_reduction_gemini_gpt:.1f}× fewer tokens',
             xy=((start3[0] + end3[0]) / 2 + 0.48, (start3[1] + end3[1]) / 2 + 2),
             ha='center', va='center', fontsize=22, fontweight='bold', color=NEUTRAL_TEXT_COLOR,
             bbox=NEUTRAL_ARROW_LABEL_BBOX,
             zorder=101)

# Add "N/A" labels for missing latency data
ax1.annotate('N/A', (x[2] - width/2, 100), ha='center', va='center', fontsize=19, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.32', fc='#F0F0F0', ec='#888888', lw=1.9, alpha=0.965),
             zorder=6)
ax1.annotate('N/A', (x[3] - width/2, 100), ha='center', va='center', fontsize=19, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.32', fc='#F0F0F0', ec='#888888', lw=1.9, alpha=0.965),
             zorder=6)

# Ensure arrows are on top by bringing them to front
ax1.set_zorder(1)  # Lower zorder for the axis
ax2.set_zorder(1)  # Lower zorder for the axis

# Layout + export
plt.tight_layout()
plt.savefig('combined_models_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('combined_models_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

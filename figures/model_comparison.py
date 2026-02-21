# performance_comparison_7b_modified.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch

# ----------------------------
# Data
# ----------------------------
models = ['Nemotron-7B', 'Sky-T1-7B']
latency_ms = [3296, 307]
avg_tokens = [9051, 494]

speedup = latency_ms[0] / latency_ms[1]          # ≈ 10.7×
token_reduction = avg_tokens[0] / avg_tokens[1]  # ≈ 18.3×

# ----------------------------
# Style and Label Colors
# ----------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.labelsize': 25,   # Moderately increased axis label size
    'axes.titlesize': 32,   # Moderately increased title size
    'legend.fontsize': 19,  # Slightly decreased legend font size
})

# Assign pale/bright colors for each model (one color per model)
MODEL_COLORS = ['#A7CDF7', '#FFE1B3']  # Light blue (Nemotron), pale orange (Sky-T1)

# Set aspect ratio to x:y = 10:12 ==> width:height = 10:12
fig, ax1 = plt.subplots(figsize=(12, 8), dpi=110)
ax2 = ax1.twinx()

# ----------------------------
# Bars (very visible, extremely sparse hatching: slashes and stars), one model = one color
# ----------------------------
width = 0.38
x = np.arange(len(models))

# Use very sparse hatching for bar differentiation (not star/o but slash/star as in code)
lat_hatch = '/'   # Sparse diagonal line hatching for latency
tok_hatch = '*'   # Sparse star hatching for token count

# Assign bar colors (one color per model: same for latency and token bar of that model)
lat_bar_colors = MODEL_COLORS
tok_bar_colors = MODEL_COLORS

# Draw latency bars with sparse slashes as hatching
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
# Draw token bars with sparse "star" hatching pattern
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
# Grid, limits, labels - grid on token axis
# ----------------------------

ax1.yaxis.grid(False)
# Remove x axis grid
ax1.xaxis.grid(False)
ax2.yaxis.grid(
    True,
    linestyle='-',
    alpha=0.40,
    zorder=5,  # above bars
    color='#AA8855',
    linewidth=1.65
)

# Adjust axis for bar/label aesthetics
ax1.set_ylim(0, 3700)
ax2.set_ylim(0, 10500)
ax1.set_xlim(-0.6, 1.6)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=24, color='#121212', fontweight='bold')
ax1.set_ylabel('Latency (ms)', fontsize=25, color='black', fontweight='bold')
ax2.set_ylabel('Decoding Token Length', fontsize=25, color='black', fontweight='bold')
ax1.set_title('Reasoning Models (7B Scale)',
              fontsize=32, pad=24, color='#232323', weight='bold', loc='center')

# Make sure grid lines also appear above all patches
for spine in ['left', 'right']:
    ax1.spines['left'].set_zorder(6)
    ax2.spines['right'].set_zorder(6)

ax1.spines['left'].set_color('black')
ax1.yaxis.label.set_color('black')
ax1.tick_params(axis='y', colors='black', labelsize=22)
ax2.spines['right'].set_color('black')
ax2.yaxis.label.set_color('black')
ax2.tick_params(axis='y', colors='black', labelsize=22)

# Simplified legend: use new hatches for latency/tokens
handles = [
    Patch(facecolor='#dddddd', edgecolor='black', hatch=lat_hatch, label='Latency (ms)', linewidth=2.7),
    Patch(facecolor='#dddddd', edgecolor='black', hatch=tok_hatch, label='Decoding Token Length', linewidth=2.7)
]
# Vertically stack the legend entries by setting ncol=1
ax1.legend(
    handles=handles,
    loc='upper right',
    frameon=True,
    fancybox=True,
    ncol=1,  # Vertical legend: one column
    handleheight=2.1,
    columnspacing=1.7,
    fontsize=19   # Slightly decreased legend font size
)

# ----------------------------
# Value labels above each bar
# ----------------------------
def add_value_labels(ax, bars, values, fmt, outline_col, bg_col):
    for b, v in zip(bars, values):
        cx = b.get_x() + b.get_width() / 2
        cy = b.get_height()
        ax.annotate(
            fmt(v), (cx, cy), xytext=(0, 13), textcoords='offset points',
            ha='center', va='bottom', fontsize=23, fontweight='bold',
            color=outline_col,
            bbox=dict(boxstyle='round,pad=0.34', fc=bg_col,
                      ec='none', lw=0.0, alpha=0.965),  # Remove annotation frame
            zorder=1  # Render beneath most elements
        )

NEUTRAL_TEXT_COLOR = '#444444'
NEUTRAL_ARROW_COLOR = '#444466'
# Remove the edge color from the arrow label bbox for frame-less annotation
NEUTRAL_ARROW_LABEL_BBOX = {'boxstyle': 'round,pad=0.4', 'fc': 'white', 'ec': 'none', 'lw': 0.0}
add_value_labels(ax1, bars_lat, latency_ms, lambda v: f'{v:,} ms', NEUTRAL_TEXT_COLOR, 'white')
add_value_labels(ax2, bars_tok, avg_tokens, lambda v: f'{v:,}', NEUTRAL_TEXT_COLOR, 'white')

# ----------------------------
# Curved arrows + callouts
# ----------------------------
lat_left_top = bars_lat[0].get_height()
lat_right_top = bars_lat[1].get_height()
tok_left_top = bars_tok[0].get_height()
tok_right_top = bars_tok[1].get_height()

latency_offset_below = 0.045 * lat_left_top
tokens_offset_below = 0.052 * tok_left_top
latency_offset_above = 0.72 * lat_right_top
tokens_offset_above = 1.15 * tok_right_top

# (A) Latency: "10.7× faster" (ax1 coordinates)
latency_center_x0 = x[0] - width / 2
latency_center_x1 = x[1] - width / 2
latency_arrow_shift = 0

start1 = (latency_center_x0 + latency_arrow_shift, lat_left_top - latency_offset_below)
end1 = (latency_center_x1 - latency_arrow_shift, lat_right_top + latency_offset_above)

# Draw arrow (latency improvement) with high zorder for visibility
arrow1 = FancyArrowPatch(
    start1, end1, transform=ax1.transData,
    connectionstyle="arc3,rad=-0.29",
    arrowstyle='->', mutation_scale=28,
    linewidth=5.5,
    color=NEUTRAL_ARROW_COLOR, zorder=100, clip_on=False
)
ax2.add_patch(arrow1)
arrow1.set_zorder(100)
ax2.annotate(
    f'{speedup:.1f}× faster',
    xy=((start1[0] + end1[0]) / 2 + 0.45, (start1[1] + end1[1]) / 2 + 1000),
    ha='center', va='center', fontsize=25, fontweight='bold', color=NEUTRAL_TEXT_COLOR,
    bbox=NEUTRAL_ARROW_LABEL_BBOX,
    zorder=1  # Render beneath most elements
)

# (B) Tokens: "18.3× fewer tokens" (ax2 coordinates)
start2 = (x[0] + width / 2, tok_left_top - tokens_offset_below)
end2 = (x[1] + width / 2, tok_right_top + tokens_offset_above)
arrow2 = FancyArrowPatch(
    start2, end2, transform=ax2.transData,
    connectionstyle="arc3,rad=-0.28",
    arrowstyle='->', mutation_scale=28,
    linewidth=5.5,
    color=NEUTRAL_ARROW_COLOR, zorder=100, clip_on=False
)
ax2.add_patch(arrow2)
arrow2.set_zorder(100)
ax2.annotate(
    f'{token_reduction:.1f}× fewer tokens',
    xy=((start2[0] + end2[0]) / 2 + 0.45, (start2[1] + end2[1]) / 2 + 2),
    ha='center', va='center', fontsize=25, fontweight='bold', color=NEUTRAL_TEXT_COLOR,
    bbox=NEUTRAL_ARROW_LABEL_BBOX,
    zorder=1  # Render beneath most elements
)

# Ensure arrows are above the axes
ax1.set_zorder(1)
ax2.set_zorder(1)

# plt.tight_layout()
fig.subplots_adjust(
    left   = 0.1,    # Make space for left y-axis label
    bottom = 0.1,    # Make space for x-axis label
    right  = 0.9,    # Make space for right y-axis label
    top    = 0.9     # Make space for the title
)
plt.savefig('performance_comparison_7b_modified.png', dpi=300, bbox_inches='tight')
plt.savefig('performance_comparison_7b_modified.pdf', dpi=300, bbox_inches='tight')  # Save to PDF as well
plt.show()
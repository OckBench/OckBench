import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ----------------------------
# Data
# ----------------------------
models = ['Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
accuracy = [16.5, 21.0, 31.0]
avg_tokens = [24207, 19381, 19098]

# ----------------------------
# Style
# ----------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.labelsize': 25,
    'axes.titlesize': 32,
    'legend.fontsize': 19,
})

# Muted pastels: teal / sage green / warm sand — same low-saturation feel
MODEL_COLORS = ['#9ECECE', '#B8D4A8', '#F0C8A0']

acc_hatch = '/'
tok_hatch = '*'

fig, ax1 = plt.subplots(figsize=(12, 8), dpi=110)
ax2 = ax1.twinx()

width = 0.34
x = np.arange(len(models))

# Accuracy bars (left axis)
bars_acc = ax1.bar(
    x - width / 2,
    accuracy,
    width,
    color=MODEL_COLORS,
    edgecolor='black',
    hatch=acc_hatch,
    linewidth=3.1,
    zorder=3,
    label='Accuracy (%)',
    alpha=1.0,
)

# Token bars (right axis)
bars_tok = ax2.bar(
    x + width / 2,
    avg_tokens,
    width,
    color=MODEL_COLORS,
    edgecolor='black',
    hatch=tok_hatch,
    linewidth=3.1,
    zorder=3,
    label='Tokens',
    alpha=1.0,
)

# ----------------------------
# Grid / limits / labels
# ----------------------------
ax1.yaxis.grid(False)
ax1.xaxis.grid(False)
ax2.yaxis.grid(
    True,
    linestyle='-',
    alpha=0.40,
    zorder=5,
    color='#AA8855',
    linewidth=1.65,
)

ax1.set_ylim(0, 45)
ax2.set_ylim(0, 32000)
ax1.set_xlim(-0.6, 2.6)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=22, color='#121212', fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=25, color='black', fontweight='bold')
ax2.set_ylabel('Tokens', fontsize=25, color='black', fontweight='bold')
ax1.set_title('Accuracy vs. Token Efficiency',
              fontsize=32, pad=24, color='#232323', weight='bold', loc='center')

for spine in ['left', 'right']:
    ax1.spines['left'].set_zorder(6)
    ax2.spines['right'].set_zorder(6)

ax1.spines['left'].set_color('black')
ax1.yaxis.label.set_color('black')
ax1.tick_params(axis='y', colors='black', labelsize=22)
ax2.spines['right'].set_color('black')
ax2.yaxis.label.set_color('black')
ax2.tick_params(axis='y', colors='black', labelsize=22)

# Legend
handles = [
    Patch(facecolor='#dddddd', edgecolor='black', hatch=acc_hatch,
          label='Accuracy (%)', linewidth=2.7),
    Patch(facecolor='#dddddd', edgecolor='black', hatch=tok_hatch,
          label='Tokens', linewidth=2.7),
]
ax1.legend(
    handles=handles,
    loc='upper left',
    frameon=True,
    fancybox=True,
    ncol=1,
    handleheight=2.1,
    columnspacing=1.7,
    fontsize=19,
)

# ----------------------------
# Value labels above bars
# ----------------------------
NEUTRAL_TEXT_COLOR = '#444444'
NEUTRAL_ARROW_COLOR = '#444466'


def add_value_labels(ax, bars, values, fmt, text_color='#444444'):
    for b, v in zip(bars, values):
        cx = b.get_x() + b.get_width() / 2
        cy = b.get_height()
        ax.annotate(
            fmt(v), (cx, cy), xytext=(0, 13), textcoords='offset points',
            ha='center', va='bottom', fontsize=21, fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.34', fc='none', ec='none', lw=0.0, alpha=0.0),
            zorder=1,
        )


add_value_labels(ax1, bars_acc, accuracy, lambda v: f'{v:.1f}%', NEUTRAL_TEXT_COLOR)
add_value_labels(ax2, bars_tok, avg_tokens, lambda v: f'{v:,}', NEUTRAL_TEXT_COLOR)

ax1.set_zorder(1)
ax2.set_zorder(1)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
plt.savefig('qwen3_series_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('qwen3_series_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

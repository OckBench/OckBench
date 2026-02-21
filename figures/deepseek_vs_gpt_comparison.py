import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch

# ----------------------------
# Data
# ----------------------------
models = ['DeepSeek-V3.2', 'GPT-5.2 (Low)']
accuracy = [43.5, 41.5]
avg_tokens = [25492, 5003]

token_reduction = avg_tokens[0] / avg_tokens[1]   # ≈ 5.1×
acc_diff = accuracy[0] - accuracy[1]               # 2.0 pp

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

MODEL_COLORS = ['#9ECECE', '#F0B0BC']   # muted teal / soft rose

acc_hatch = '/'
tok_hatch = '*'

fig, ax1 = plt.subplots(figsize=(12, 8), dpi=110)
ax2 = ax1.twinx()

width = 0.38
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

ax1.set_ylim(0, 55)
ax2.set_ylim(0, 32000)
ax1.set_xlim(-0.6, 1.6)
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
    loc='upper right',
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
NEUTRAL_ARROW_LABEL_BBOX = {'boxstyle': 'round,pad=0.4', 'fc': 'white', 'ec': 'none', 'lw': 0.0}


def add_value_labels(ax, bars, values, fmt, text_color='#444444'):
    for b, v in zip(bars, values):
        cx = b.get_x() + b.get_width() / 2
        cy = b.get_height()
        ax.annotate(
            fmt(v), (cx, cy), xytext=(0, 13), textcoords='offset points',
            ha='center', va='bottom', fontsize=23, fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.34', fc='white', ec='none', lw=0.0, alpha=0.965),
            zorder=1,
        )


add_value_labels(ax1, bars_acc, accuracy, lambda v: f'{v:.1f}%', NEUTRAL_TEXT_COLOR)
add_value_labels(ax2, bars_tok, avg_tokens, lambda v: f'{v:,}', NEUTRAL_TEXT_COLOR)

# ----------------------------
# Curved arrow: token reduction
# ----------------------------
tok_left_top = bars_tok[0].get_height()
tok_right_top = bars_tok[1].get_height()

start_tok = (x[0] + width / 2, tok_left_top - 0.05 * tok_left_top)
end_tok   = (x[1] + width / 2, tok_right_top + 0.55 * tok_right_top)

arrow_tok = FancyArrowPatch(
    start_tok, end_tok, transform=ax2.transData,
    connectionstyle='arc3,rad=-0.28',
    arrowstyle='->', mutation_scale=28,
    linewidth=5.5,
    color=NEUTRAL_ARROW_COLOR, zorder=100, clip_on=False,
)
ax2.add_patch(arrow_tok)
arrow_tok.set_zorder(100)

ax1.set_zorder(1)
ax2.set_zorder(1)

# Text in upper axes space, above the bars
ax2.text(
    0.5, 0.97,
    f'{token_reduction:.1f}× fewer tokens',
    transform=ax2.transAxes,
    ha='center', va='top',
    fontsize=20, fontweight='bold',
    color=NEUTRAL_ARROW_COLOR,
)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
plt.savefig('deepseek_vs_gpt_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('deepseek_vs_gpt_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------
# 1. Raw data
# --------------------------------------------------
data = [
    ("GPT-4o", "Commercial", 530.02, 61.00, 868.9),
    ("Sky-T1-7B", "Open Source", 511.45, 58.00, 881.8),         # will be removed
    ("GPT-4.1", "Commercial", 977.43, 80.50, 1214.2),
    ("Qwen3-235B-A22B-non_thinking", "Open Source", 963.65, 62.00, 1554.3),
    ("GPT-5", "Commercial", 2016.37, 88.50, 2278.4),
    ("GPT-o3", "Commercial", 2280.02, 83.00, 2747.0),
    ("Qwen3-32B-non_thinking", "Open Source", 1695.11, 58.50, 2897.6),
    ("Qwen3-30B-A3B-non_thinking", "Open Source", 1839.41, 58.00, 3171.4),
    ("Qwen3-14B-non_thinking", "Open Source", 2406.07, 60.00, 4010.1),
    ("Qwen3-8B-non_thinking", "Open Source", 2188.66, 52.50, 4168.9),
    ("Gemini-2.5 Pro", "Commercial", 4011.93, 87.50, 4585.1),
    ("Qwen3-4B-non_thinking", "Open Source", 2247.39, 49.00, 4586.5),
    ("Gemini-2.5 Flash", "Commercial", 3970.64, 85.00, 4671.3),
    ("AceReason-Nemotron-14B", "Open Source", 4028.84, 71.50, 5634.7),
    # ("Qwen3-32B-thinking", "Open Source", 5217.71, 73.50, 7098.9),
    ("Qwen3-30B-A3B-thinking", "Open Source", 5399.09, 74.50, 7247.1),
    ("AceReason-Nemotron-7B", "Open Source", 4271.45, 57.50, 7428.6),
    ("AReaL-boba-2-32B", "Open Source", 5710.06, 76.00, 7513.2),
    # ("Qwen3-14B-thinking", "Open Source", 5775.52, 73.50, 7857.9),
    ("AReal-boba-2-14B", "Open Source", 7133.56, 74.50, 9575.2),
    ("Qwen3-235B-A22B-thinking", "Open Source", 7254.51, 71.50, 10146.2),
    # ("Qwen3-8B-thinking", "Open Source", 7321.98, 70.00, 10460.0),
    ("AReal-boba-2-8B", "Open Source", 8684.22, 67.00, 12961.5),
    ("Qwen3-4B-thinking", "Open Source", 8942.01, 64.00, 13971.9),
    ("Sky-T1-mini", "Open Source", 8651.71, 50.50, 17132.1),    # will be removed
]

df = pd.DataFrame(
    data,
    columns=["Model", "Category", "Tokens", "Accuracy", "Efficiency"],
)
df["Model"] = df["Model"].str.strip()

# Remove Sky models
df = df[~df["Model"].isin(["Sky-T1-7B", "Sky-T1-mini"])].copy()

# --------------------------------------------------
# 2. Family, parameter size, marker size
# --------------------------------------------------
def get_family(name: str) -> str:
    if name.startswith("GPT-"):
        return "GPT"
    if name.startswith("Gemini-2.5"):
        return "Gemini"
    if name.startswith("Qwen3"):
        return "Qwen3"
    if "Nemotron" in name:
        return "AceReason-Nemotron"
    if "AReaL" in name or "AReal" in name:
        return "AReaL-boba"
    return "Other"

df["Family"] = df["Model"].apply(get_family)

def parse_params_B(name: str):
    # Match pieces like "14B", "235B", "4B", etc.
    matches = re.findall(r"(\d+)\s*B", name)
    if not matches:
        return np.nan
    return max(float(m) for m in matches)

df["ParamsB"] = df["Model"].apply(parse_params_B)
df["SizeKnown"] = df["ParamsB"].notna()

# Marker area from ParamsB (log scale) for models with known size
valid = df["ParamsB"].dropna()
logp = np.log10(valid)
log_min, log_max = logp.min(), logp.max()

# Larger sizes for a large 24x10 figure
size_min, size_max = 200, 1200  # points^2

def compute_size(params_B):
    if np.isnan(params_B):
        return np.nan
    if log_max == log_min:
        return (size_min + size_max) / 2.0
    v = np.log10(params_B)
    return size_min + (v - log_min) / (log_max - log_min) * (size_max - size_min)

df["MarkerSize"] = df["ParamsB"].apply(compute_size)

# For unknown sizes, use a large but not extreme bubble size
known_sizes = df.loc[df["SizeKnown"], "MarkerSize"]
estimated_large_size = np.percentile(known_sizes, 75) * 3  # upper quartile of known sizes
df.loc[~df["SizeKnown"], "MarkerSize"] = estimated_large_size

# --------------------------------------------------
# 3. Color scheme by family
# --------------------------------------------------
FAMILY_COLORS = {
    "GPT": "#B2DFDB",               # pale teal / green
    "Gemini": "#D1C4E9",            # pale purple
    "Qwen3": "#FFE0B2",             # pale orange
    "AceReason-Nemotron": "#BBDEFB",# pale blue
    "AReaL-boba": "#FFCCBC",        # pale peach
    "Other": "#E0E0E0",
}

# Shorter labels for some models
label_map = {
    "Qwen3-235B-A22B-non_thinking": "Qwen3-235B",
    "Qwen3-235B-A22B-thinking": "Qwen3-235B T",
    "Qwen3-30B-A3B-non_thinking": "Qwen3-30B",
    "Qwen3-30B-A3B-thinking": "Qwen3-30B T",
    "Qwen3-32B-non_thinking": "Qwen3-32B",
    "Qwen3-32B-thinking": "Qwen3-32B T",
    "Qwen3-14B-non_thinking": "Qwen3-14B",
    "Qwen3-14B-thinking": "Qwen3-14B T",
    "Qwen3-8B-non_thinking": "Qwen3-8B",
    "Qwen3-8B-thinking": "Qwen3-8B T",
    "Qwen3-4B-non_thinking": "Qwen3-4B",
    "Qwen3-4B-thinking": "Qwen3-4B T",
    "AceReason-Nemotron-14B": "Nemotron-14B",
    "AceReason-Nemotron-7B": "Nemotron-7B",
    "AReaL-boba-2-32B": "AReaL-32B",
    "AReal-boba-2-14B": "AReaL-14B",
    "AReal-boba-2-8B": "AReaL-8B",
}

# Offsets for labels that tend to collide; you can extend this dictionary
label_offsets = {
    "Gemini-2.5 Pro": (0, 20),
    "Gemini-2.5 Flash": (0, -45),
    "GPT-5": (-60, 15),
    "GPT-o3": (0, -50),
    "GPT-4o": (0, 36),
    "GPT-4.1": (0, 36),
    "Qwen3-235B-A22B-non_thinking": (0, 20),
    "Qwen3-30B-A3B-thinking": (0, -20),
    "Qwen3-30B-A3B-non_thinking": (0, -30),
    "Nemotron-14B": (-20, 0),
}

# --------------------------------------------------
# 4. Plot
# --------------------------------------------------
plt.close("all")
fig, ax = plt.subplots(figsize=(24, 10), dpi=300)

# Scatter by family; dotted border if size is hidden (commercial, no ParamsB)
for family, group in df.groupby("Family"):
    color = FAMILY_COLORS.get(family, "#E0E0E0")

    g_known = group[group["SizeKnown"]]
    if not g_known.empty:
        ax.scatter(
            g_known["Tokens"],
            g_known["Accuracy"],
            s=g_known["MarkerSize"],
            facecolors=color,
            edgecolors="black",
            linewidths=1.2,
            alpha=0.9,
        )

    g_unknown = group[~group["SizeKnown"]]
    if not g_unknown.empty:
        ax.scatter(
            g_unknown["Tokens"],
            g_unknown["Accuracy"],
            s=g_unknown["MarkerSize"],
            facecolors=color,
            edgecolors="black",
            linewidths=1.2,
            alpha=0.9,
            linestyle="dotted",  # dotted border for hidden size
        )

# Labels for all models
for _, row in df.iterrows():
    label = label_map.get(row["Model"], row["Model"])
    dx, dy = label_offsets.get(row["Model"], (0, 12))
    ax.annotate(
        label,
        (row["Tokens"], row["Accuracy"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="center",
        fontsize=20,
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec="none",
            alpha=0,  # fully transparent annotation background
        ),
    )

# Axes, grid, limits
ax.set_xscale("log")
ax.set_xlabel("# Tokens (log scale)", fontsize=36)
ax.set_ylabel("Accuracy (%)", fontsize=36)
# ax.set_title(
#     "Model Performance: Accuracy vs. Decoding Token Length",
#     fontsize=40,
#     pad=16,
# )

ax.tick_params(axis="both", which="major", labelsize=16)
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)

x_min = df["Tokens"].min() * 0.7
x_max = df["Tokens"].max() * 1.3
y_min = df["Accuracy"].min() - 3
y_max = df["Accuracy"].max() + 3
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Legends: families (color) in upper left, border style in lower right
families_present = df["Family"].unique()
family_handles = [
    Patch(
        facecolor=FAMILY_COLORS[fam],
        edgecolor="black",
        label=fam,
    )
    for fam in families_present
]

size_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="solid",
        markerfacecolor="white",
        markeredgecolor="black",
        label="Model size known",
        markersize=12,
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="dotted",
        markerfacecolor="white",
        markeredgecolor="black",
        label="Commercial (size hidden)",
        markersize=12,
    ),
]

legend1 = ax.legend(
    handles=family_handles,
    title="Model family",
    loc="upper left",
    fontsize=20,
    title_fontsize=25,
)
ax.add_artist(legend1)

legend2 = ax.legend(
    handles=size_handles,
    title="Marker border",
    loc="lower right",
    fontsize=20,
    title_fontsize=25,
)

fig.tight_layout()

fig.savefig("reasoning_efficiency_scatter_v3.pdf")
fig.savefig("reasoning_efficiency_scatter_v3.png", dpi=300)

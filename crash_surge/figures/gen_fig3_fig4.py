"""Generate Figure 3 (contagion heatmap) and Figure 4 (recovery vs persistence)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 9

DATA_DIR = str(Path(__file__).resolve().parents[1] / "results")
OUT_DIR = str(Path(__file__).resolve().parents[2] / "figures")
Path(OUT_DIR).mkdir(exist_ok=True)

CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
LABELS = ["Fam", "SP", "Mem", "Conv", "Enj"]

with open(f"{DATA_DIR}/exp2_contagion_recovery.json") as f:
    crash_data = json.load(f)
with open(f"{DATA_DIR}/exp20_surge_contagion_recovery.json") as f:
    surge_data = json.load(f)

# --- Figure 3: Dual Contagion Heatmap ---
def build_matrix(d):
    mat = np.zeros((5, 5))
    for i, ci in enumerate(CONSTRUCTS):
        for j, cj in enumerate(CONSTRUCTS):
            mat[i, j] = d[ci][cj]
    return mat

crash_mat = build_matrix(crash_data["cooccurrence_matrix"])
surge_mat = build_matrix(surge_data["surge_cooccurrence"])

# Find shared color scale (excluding diagonal)
mask = ~np.eye(5, dtype=bool)
vmin = 0
vmax = max(crash_mat[mask].max(), surge_mat[mask].max())
vmax = np.ceil(vmax * 10) / 10  # round up to nearest 0.1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

for ax, mat, title in [(ax1, crash_mat, "Crash Co-occurrence"),
                        (ax2, surge_mat, "Surge Co-occurrence")]:
    im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    for i in range(5):
        for j in range(5):
            val = mat[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

fig.subplots_adjust(right=0.88, wspace=0.25)
cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
fig.colorbar(im, cax=cbar_ax, label="Co-occurrence Rate")

fig.savefig(f"{OUT_DIR}/fig_contagion_heatmap.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved fig_contagion_heatmap.pdf")

# --- Figure 4: Recovery vs Persistence Bar Chart ---
recovery = crash_data["recovery_summary"]
persistence = surge_data["surge_persistence"]

recovery_rates = [recovery[c]["recovery_rate"] * 100 for c in CONSTRUCTS]
persistence_rates = [persistence[c]["persistence_rate"] * 100 for c in CONSTRUCTS]

x = np.arange(len(CONSTRUCTS))
width = 0.32

fig, ax = plt.subplots(figsize=(5.0, 3.0))
bars1 = ax.bar(x - width/2, recovery_rates, width, label="Crash Recovery",
               color="#d95f02", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, persistence_rates, width, label="Surge Persistence",
               color="#1b9e77", edgecolor="white", linewidth=0.5)

ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, zorder=0)
ax.set_ylabel("Rate (%)", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylim(0, 100)
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=7)

fig.savefig(f"{OUT_DIR}/fig_recovery_persistence.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved fig_recovery_persistence.pdf")

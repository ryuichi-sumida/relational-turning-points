"""Figure 6: Forecast vs Detection Comparison (Crashes and Surges side by side).

Values are max best_AUPRC across 7 feature sets from ablation studies.
Detection: results/ablation/summary.csv
Forecasting: results/ablation_forecast/summary.csv

Filled stars = forecasting significantly > detection (p<.05, paired bootstrap).
Open stars = detection significantly > forecasting (p<.05, paired bootstrap).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

construct_labels = ['Fam', 'SP', 'Mem', 'Conv', 'Enj']
# Order: familiarity, social_penetration, memory, conversational, enjoyment

# --- CRASH data (max best_AUPRC across 7 feature sets) ---
# Detection ablation
crash_det = [0.2069, 0.1482, 0.1632, 0.2490, 0.1491]
# Forecast ablation
crash_fore = [0.3560, 0.2253, 0.1453, 0.1870, 0.1938]

# --- SURGE data (max best_AUPRC across 7 feature sets) ---
# Detection ablation
surge_det = [0.2476, 0.2835, 0.1995, 0.3757, 0.2218]
# Forecast ablation
surge_fore = [0.3011, 0.2011, 0.2590, 0.3945, 0.2310]

# --- Significance (independent bootstrap, p<.05, exp27) ---
# Order: familiarity, social_penetration, memory, conversational, enjoyment
# Crash: SP forecast sig > detection; Enj forecast sig > detection (EN model)
# Conv detection > forecast (borderline, CI upper=0.0018 — not sig at p<.05)
# Surge: No significant differences in either direction
crash_fore_sig_better = [False, True, False, False, True]   # forecast > detection significant
crash_det_sig_better  = [False, False, False, False, False]  # detection > forecast significant
surge_fore_sig_better = [False, False, False, False, False]  # forecast > detection significant
surge_det_sig_better  = [False, False, False, False, False]  # detection > forecast significant

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)

x = np.arange(len(construct_labels))
bw = 0.32
gap = 0.04

crash_det_color = '#D9534F'
crash_fore_color = '#F5A6A3'
surge_det_color = '#2A7B9B'
surge_fore_color = '#88CED8'

# Left panel: Crashes
ax1.bar(x - bw/2 - gap/2, crash_det, bw,
        color=crash_det_color, edgecolor='white', linewidth=0.5,
        label='Detection', zorder=3)
ax1.bar(x + bw/2 + gap/2, crash_fore, bw,
        color=crash_fore_color, edgecolor=crash_det_color, linewidth=0.8,
        hatch='///', label='Forecast', zorder=3)

# Significance markers for crashes
for i in range(len(construct_labels)):
    top = max(crash_det[i], crash_fore[i])
    if crash_fore_sig_better[i]:
        # Filled star: forecasting significantly better
        ax1.annotate('\u2605', xy=(x[i], top + 0.01),
                     fontsize=8, ha='center', va='bottom', color='#333333')
    elif crash_det_sig_better[i]:
        # Open star: detection significantly better
        ax1.annotate('\u2606', xy=(x[i], top + 0.01),
                     fontsize=8, ha='center', va='bottom', color='#333333')

ax1.set_title('Crashes', fontsize=9, fontweight='bold')
ax1.set_ylabel('AUPRC', fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(construct_labels, fontsize=7)
ax1.legend(fontsize=6.5, loc='upper right', framealpha=0.9, edgecolor='#cccccc')

# Right panel: Surges
ax2.bar(x - bw/2 - gap/2, surge_det, bw,
        color=surge_det_color, edgecolor='white', linewidth=0.5,
        label='Detection', zorder=3)
ax2.bar(x + bw/2 + gap/2, surge_fore, bw,
        color=surge_fore_color, edgecolor=surge_det_color, linewidth=0.8,
        hatch='///', label='Forecast', zorder=3)

# Significance markers for surges
for i in range(len(construct_labels)):
    top = max(surge_det[i], surge_fore[i])
    if surge_fore_sig_better[i]:
        # Filled star: forecasting significantly better
        ax2.annotate('\u2605', xy=(x[i], top + 0.01),
                     fontsize=8, ha='center', va='bottom', color='#333333')
    elif surge_det_sig_better[i]:
        # Open star: detection significantly better
        ax2.annotate('\u2606', xy=(x[i], top + 0.01),
                     fontsize=8, ha='center', va='bottom', color='#333333')

ax2.set_title('Surges', fontsize=9, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(construct_labels, fontsize=7)
ax2.legend(fontsize=6.5, loc='upper right', framealpha=0.9, edgecolor='#cccccc')

for ax in [ax1, ax2]:
    ax.set_ylim(0, 0.7)
    ax.set_xlim(-0.6, len(construct_labels) - 0.4)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, zorder=0)

plt.tight_layout(pad=0.5)
outpath = Path(__file__).resolve().parents[2] / 'figures' / 'fig_forecast_vs_detection.pdf'
outpath.parent.mkdir(exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'Saved Figure 6 to {outpath}')

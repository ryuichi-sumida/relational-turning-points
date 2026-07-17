"""Figure: Detection AUPRC Bar Chart (Crash vs Surge) — EN+STP primary config.

Uses exp28b bootstrap results (correct pipeline: stability selection + nested CV).
* marks AUPRC significantly above respective base rate.
† marks significant surge–crash differences.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# ---------- Load exp28b results (EN+STP with stability selection) ----------
exp28b_path = Path(__file__).resolve().parents[1] / 'results' / 'exp28b_en_stp_bootstrap.json'
with open(exp28b_path) as f:
    data = json.load(f)

# ---------- Construct order (matches paper) ----------
constructs = ['familiarity', 'social_penetration', 'memory',
              'conversational', 'enjoyment', 'systemic']
construct_labels = ['Familiarity', 'Social\nPenetration', 'Memory',
                    'Conv.\nQuality', 'Enjoyment', 'Systemic\n(2+)']

# Extract values from exp28b
crash_auprc = []
surge_auprc = []
crash_base = []
surge_base = []
crash_ci_lo = []
crash_ci_hi = []
surge_ci_lo = []
surge_ci_hi = []
crash_sig_vs_base = []
surge_sig_vs_base = []
surge_gt_crash_sig = []

for c in constructs:
    t = data['per_target'][c]
    crash_auprc.append(t['crash']['AUPRC'])
    surge_auprc.append(t['surge']['AUPRC'])
    crash_base.append(t['crash_rate'])
    surge_base.append(t['surge_rate'])
    crash_ci_lo.append(t['crash']['CI_lower'])
    crash_ci_hi.append(t['crash']['CI_upper'])
    surge_ci_lo.append(t['surge']['CI_lower'])
    surge_ci_hi.append(t['surge']['CI_upper'])
    crash_sig_vs_base.append(t['crash']['vs_base_rate']['significant'])
    surge_sig_vs_base.append(t['surge']['vs_base_rate']['significant'])
    surge_gt_crash_sig.append(t['paired_comparison']['surge_significantly_better'])

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(3.5, 2.8))

x = np.arange(len(construct_labels))
bar_width = 0.32
gap = 0.04

crash_color = '#D9534F'
surge_color = '#5BC0BE'

# Error bars (yerr needs [lower_err, upper_err])
crash_yerr = [[max(0, crash_auprc[i] - crash_ci_lo[i]) for i in range(len(constructs))],
              [max(0, crash_ci_hi[i] - crash_auprc[i]) for i in range(len(constructs))]]
surge_yerr = [[max(0, surge_auprc[i] - surge_ci_lo[i]) for i in range(len(constructs))],
              [max(0, surge_ci_hi[i] - surge_auprc[i]) for i in range(len(constructs))]]

ax.bar(x - bar_width/2 - gap/2, crash_auprc, bar_width,
       yerr=crash_yerr, capsize=2, error_kw={'linewidth': 0.7},
       color=crash_color, edgecolor='white', linewidth=0.5,
       label='Crash', zorder=3)

ax.bar(x + bar_width/2 + gap/2, surge_auprc, bar_width,
       yerr=surge_yerr, capsize=2, error_kw={'linewidth': 0.7},
       color=surge_color, edgecolor='white', linewidth=0.5,
       label='Surge', zorder=3)

# Separate base rate lines for crash and surge
for i in range(len(constructs)):
    # Crash base rate line (red, dashed)
    ax.plot([x[i] - bar_width - gap/2 - 0.02, x[i] - 0.02],
            [crash_base[i], crash_base[i]],
            color=crash_color, linewidth=0.8, linestyle='--', alpha=0.6, zorder=2)
    # Surge base rate line (teal, dashed)
    ax.plot([x[i] + 0.02, x[i] + bar_width + gap/2 + 0.02],
            [surge_base[i], surge_base[i]],
            color=surge_color, linewidth=0.8, linestyle='--', alpha=0.6, zorder=2)

# Legend entries for base rates
ax.plot([], [], color=crash_color, linewidth=0.8, linestyle='--', alpha=0.6, label='Crash base rate')
ax.plot([], [], color=surge_color, linewidth=0.8, linestyle='--', alpha=0.6, label='Surge base rate')

# Significance markers: † for surge > crash
for i in range(len(constructs)):
    if surge_gt_crash_sig[i]:
        top = max(crash_auprc[i] + crash_yerr[1][i],
                  surge_auprc[i] + surge_yerr[1][i]) + 0.025
        ax.annotate('\u2020', xy=(x[i], top),
                    fontsize=7, ha='center', va='bottom', color='#333333')

ax.set_ylabel('AUPRC', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(construct_labels, fontsize=6.5)
ax.set_ylim(0, 0.55)
ax.set_xlim(-0.6, len(construct_labels) - 0.4)
ax.tick_params(axis='y', labelsize=7)
ax.legend(fontsize=5.5, loc='upper left', framealpha=0.9, edgecolor='#cccccc',
          ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)
ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, zorder=0)

plt.tight_layout(pad=0.4)
outpath = Path(__file__).resolve().parents[2] / 'figures' / 'fig_detection_auprc.pdf'
outpath.parent.mkdir(exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'Saved figure to {outpath}')

# Print summary for verification
print('\nexp28b EN+STP Detection AUPRC:')
for i, c in enumerate(constructs):
    flags = []
    if crash_sig_vs_base[i]: flags.append('crash>base*')
    if surge_sig_vs_base[i]: flags.append('surge>base*')
    if surge_gt_crash_sig[i]: flags.append('surge>crash†')
    print(f'  {construct_labels[i].replace(chr(10)," "):20s}  '
          f'crash={crash_auprc[i]:.3f} [{crash_ci_lo[i]:.3f},{crash_ci_hi[i]:.3f}]  '
          f'surge={surge_auprc[i]:.3f} [{surge_ci_lo[i]:.3f},{surge_ci_hi[i]:.3f}]  '
          f'{" ".join(flags)}')

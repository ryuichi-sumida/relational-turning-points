"""
Mediation path diagram: Memory(t) → Social Penetration(t+1) → Enjoyment(t+1)
Shows paths a, b, c' with coefficients. Single-column width for ACM acmart.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Path coefficients from mediation analysis ─────────────────────────────
a_coef, a_p = 0.165, 0.001
b_coef, b_p = 0.497, 0.0001
cprime_coef, cprime_p = 0.005, 0.943
indirect = 0.082
ci_lo, ci_hi = 0.029, 0.155

# ── Styling ───────────────────────────────────────────────────────────────
SIG_COLOR = '#2E86AB'
NONSIG_COLOR = '#999999'
NODE_COLOR = '#F0F0F0'
NODE_EDGE = '#333333'
INDIRECT_COLOR = '#2E86AB'

fig, ax = plt.subplots(1, 1, figsize=(3.33, 2.0))

# ── Node positions ────────────────────────────────────────────────────────
# Triangle layout: X (left), M (top-right), Y (bottom-right)
nodes = {
    'X': (0.15, 0.55, 'Perceived\nMemory$(t)$'),
    'M': (0.85, 0.85, 'Social\nPenetration$(t{+}1)$'),
    'Y': (0.85, 0.25, 'Enjoyment$(t{+}1)$'),
}

box_w, box_h = 0.28, 0.22

for key, (cx, cy, label) in nodes.items():
    rect = mpatches.FancyBboxPatch(
        (cx - box_w/2, cy - box_h/2), box_w, box_h,
        boxstyle='round,pad=0.02',
        facecolor=NODE_COLOR, edgecolor=NODE_EDGE, linewidth=0.8, zorder=3
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=5.5, fontweight='bold', zorder=4)

# ── Draw arrows ───────────────────────────────────────────────────────────

def draw_arrow(ax, start, end, label, color, lw, style='->', linestyle='-',
               label_offset=(0, 0), fontsize=6):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               linestyle=linestyle),
                zorder=2)
    mid_x = (start[0] + end[0]) / 2 + label_offset[0]
    mid_y = (start[1] + end[1]) / 2 + label_offset[1]
    ax.text(mid_x, mid_y, label, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight='bold', zorder=5,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                     edgecolor='none', alpha=0.95))

# Path a: X → M (significant)
stars_a = '**' if a_p < 0.01 else '*'
draw_arrow(ax,
           (0.15 + box_w/2, 0.55 + box_h/2 * 0.3),
           (0.85 - box_w/2, 0.85 - box_h/2 * 0.3),
           f'$a = {a_coef:.3f}${stars_a}',
           SIG_COLOR, 1.2,
           label_offset=(0, 0.06))

# Path b: M → Y (significant)
stars_b = '***' if b_p < 0.001 else '**'
draw_arrow(ax,
           (0.85, 0.85 - box_h/2),
           (0.85, 0.25 + box_h/2),
           f'$b = {b_coef:.3f}${stars_b}',
           SIG_COLOR, 1.2,
           label_offset=(0.16, 0))

# Path c': X → Y (not significant, dashed)
draw_arrow(ax,
           (0.15 + box_w/2, 0.55 - box_h/2 * 0.3),
           (0.85 - box_w/2, 0.25 + box_h/2 * 0.3),
           f"$c' = {cprime_coef:.3f}$ n.s.",
           NONSIG_COLOR, 0.8, linestyle='--',
           label_offset=(0, -0.06))

# ── Indirect effect annotation ────────────────────────────────────────────
indirect_text = (f'Indirect: $a \\times b = {indirect:.3f}$\n'
                 f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')
ax.text(0.50, 0.02, indirect_text, ha='center', va='bottom',
        fontsize=5.5, color=INDIRECT_COLOR, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#E8F4FD',
                 edgecolor=INDIRECT_COLOR, alpha=0.8, linewidth=0.5))

# ── Axis setup ────────────────────────────────────────────────────────────
ax.set_xlim(-0.05, 1.15)
ax.set_ylim(-0.08, 1.05)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0.2)

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = str(Path(__file__).resolve().parents[1] / 'figures')
Path(out_dir).mkdir(exist_ok=True)
plt.savefig(f'{out_dir}/fig_mediation.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/fig_mediation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved to {out_dir}/fig_mediation.pdf')

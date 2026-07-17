"""
Concurrent (same-session) regression for all 5 constructs.
For each construct Y, fit: Y ~ other 4 constructs + session_c + C(user_id)
with cluster-robust SEs. Creates a concurrent path diagram.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "user_assessment_labels.csv"
OUT_DIR = Path(__file__).resolve().parent

CONSTRUCTS = [
    ("w1", "familiarity",        "Familiarity"),
    ("w2", "social_penetration", "Social Penetration"),
    ("w3", "memory",             "Perceived Memory"),
    ("w4", "conversational",     "Conv. Quality"),
    ("w5", "enjoyment",          "Enjoyment"),
]

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
for w_col, csv_col, _ in CONSTRUCTS:
    df[w_col] = df[csv_col]

df = df.sort_values(["user_id", "session"]).reset_index(drop=True)
df["session_c"] = df["session"] - df["session"].mean()

w_cols = [c[0] for c in CONSTRUCTS]
print(f"Data: {len(df)} obs, {df['user_id'].nunique()} participants\n")

# ── Run concurrent regressions ────────────────────────────────────────────
rows = []
for out_w, out_csv, out_label in CONSTRUCTS:
    predictors = [c[0] for c in CONSTRUCTS if c[0] != out_w]
    pred_str = " + ".join(predictors)
    formula = f"{out_w} ~ {pred_str} + session_c + C(user_id)"

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["user_id"]}
    )

    print(f"=== {out_label} (R² = {model.rsquared:.3f}) ===")
    for pred_w, pred_csv, pred_label in CONSTRUCTS:
        if pred_w == out_w:
            continue
        beta = float(model.params[pred_w])
        se = float(model.bse[pred_w])
        p = float(model.pvalues[pred_w])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {pred_label:20s} β={beta:.3f}  SE={se:.3f}  p={p:.4f} {sig}")
        rows.append({
            "predictor": pred_w,
            "predictor_label": pred_label,
            "outcome": out_w,
            "outcome_label": out_label,
            "beta": beta,
            "se": se,
            "p_value": p,
            "r2": float(model.rsquared),
            "n_obs": int(model.nobs),
        })
    print()

conc_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "analysis" / "concurrent_regression_all.csv"
conc_df.to_csv(csv_path, index=False)

json_path = OUT_DIR / "analysis" / "concurrent_regression_all.json"
sig_paths = conc_df[conc_df["p_value"] < 0.05].sort_values("p_value")
results_json = {
    "model": "Y ~ other_4_constructs + session_c + C(user_id)",
    "standard_errors": "cluster-robust (grouped by user_id)",
    "n_obs": int(df.shape[0]),
    "significant_paths": [
        {
            "predictor": r["predictor_label"],
            "outcome": r["outcome_label"],
            "beta": round(r["beta"], 3),
            "p": round(r["p_value"], 4),
        }
        for _, r in sig_paths.iterrows()
    ],
}
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)

print(f"Saved CSV to {csv_path}")
print(f"Saved JSON to {json_path}")

# ══════════════════════════════════════════════════════════════════════════
# CONCURRENT PATH DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

constructs_layout = [
    ("w1", "Familiarity"),
    ("w2", "Social\nPenetration"),
    ("w3", "Perceived\nMemory"),
    ("w4", "Conv.\nQuality"),
    ("w5", "Enjoyment"),
]

n = len(constructs_layout)
angle_offset = np.pi / 2
angles = [angle_offset - 2 * np.pi * i / n for i in range(n)]
radius = 1.05
positions = {c[0]: (radius * np.cos(a), radius * np.sin(a))
             for c, a in zip(constructs_layout, angles)}

SIG_COLOR = '#2E86AB'
NONSIG_COLOR = '#CCCCCC'
NODE_COLOR = '#F0F0F0'
NODE_EDGE = '#333333'

TITLE_SIZE = 8
NODE_SIZE = 6
ARROW_LABEL_SIZE = 5.5
LEGEND_SIZE = 6

fig, ax = plt.subplots(1, 1, figsize=(3.33, 3.3))

node_radius = 0.42
for key, label in constructs_layout:
    x, y = positions[key]
    circle = plt.Circle((x, y), node_radius, facecolor=NODE_COLOR,
                        edgecolor=NODE_EDGE, linewidth=1.0, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=NODE_SIZE, fontweight='bold', zorder=4)

# Draw all paths (undirected — concurrent association)
# For concurrent, we draw lines (not arrows) since there's no directionality
# But for clarity with multiple predictors, we use thin bidirectional arrows
drawn_pairs = set()
for _, row in conc_df.iterrows():
    pred = row['predictor']
    outcome = row['outcome']
    beta = row['beta']
    p = row['p_value']
    sig = p < 0.05

    # Only draw each pair once (use the one with the larger |beta|)
    pair_key = tuple(sorted([pred, outcome]))
    if pair_key in drawn_pairs:
        continue

    # Find the reverse path
    reverse = conc_df[(conc_df['predictor'] == outcome) & (conc_df['outcome'] == pred)]
    if len(reverse) > 0:
        rev_beta = reverse.iloc[0]['beta']
        rev_p = reverse.iloc[0]['p_value']
        # Use the path with larger absolute beta for the label
        if abs(rev_beta) > abs(beta):
            beta, p, sig = rev_beta, rev_p, rev_p < 0.05
            # swap pred/outcome for arrow direction
            pred, outcome = outcome, pred

    drawn_pairs.add(pair_key)

    x1, y1 = positions[pred]
    x2, y2 = positions[outcome]

    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist

    start_x = x1 + ux * (node_radius + 0.02)
    start_y = y1 + uy * (node_radius + 0.02)
    end_x = x2 - ux * (node_radius + 0.05)
    end_y = y2 - uy * (node_radius + 0.05)

    # Small perpendicular offset to avoid overlap
    perp_x, perp_y = -uy * 0.02, ux * 0.02
    start_x += perp_x
    start_y += perp_y
    end_x += perp_x
    end_y += perp_y

    if sig:
        color = SIG_COLOR
        lw = 1.2
        alpha = 0.9
    else:
        color = NONSIG_COLOR
        lw = 0.5
        alpha = 0.3

    # Use double-headed arrow for concurrent (non-directional)
    # Increase arc radius for adjacent nodes so the line isn't hidden behind circles
    edge_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    arc_rad = 0.20 if edge_dist < 1.0 else 0.08
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='-', color=color,
                               lw=lw, alpha=alpha,
                               connectionstyle=f'arc3,rad={arc_rad}'),
                zorder=2)

    # Coefficient labels removed — values reported in Appendix table

ax.set_xlim(-1.85, 1.85)
ax.set_ylim(-1.75, 1.85)
ax.set_aspect('equal')
ax.axis('off')

legend_elements = [
    plt.Line2D([0], [0], color=SIG_COLOR, lw=1.2, label='$p < .05$'),
    plt.Line2D([0], [0], color=NONSIG_COLOR, lw=0.5, alpha=0.5, label='n.s.'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=LEGEND_SIZE,
          frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
          bbox_to_anchor=(0.98, 0.02))

ax.set_title('Concurrent Associations (same session)',
             fontsize=TITLE_SIZE, fontweight='bold', pad=5)

plt.tight_layout()

pdf_path = OUT_DIR / "figures" / "fig_concurrent_paths.pdf"
png_path = OUT_DIR / "figures" / "fig_concurrent_paths.png"
plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nConcurrent path diagram saved to:\n  {pdf_path}\n  {png_path}")

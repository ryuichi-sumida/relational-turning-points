"""
Re-run traditional 5x5 CLPM and regenerate path diagram for Paper_ICMI_v2.
Model: outcome(t+1) ~ session_c + predictor(t) + outcome(t) + C(user_id)
       with cluster-robust SEs grouped by user_id.
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
    ("w1_familiarity",  "familiarity",          "Familiarity"),
    ("w2_penetration",  "social_penetration",   "Social Penetration"),
    ("w3_memory",       "memory",               "Perceived Memory"),
    ("w4_quality",      "conversational",       "Conv. Quality"),
    ("w5_enjoy_intent", "enjoyment",            "Enjoyment"),
]

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Map column names to w-notation
col_map = {csv_col: w_col for w_col, csv_col, _ in CONSTRUCTS}
for w_col, csv_col, _ in CONSTRUCTS:
    if csv_col in df.columns:
        df[w_col] = df[csv_col]

df = df.sort_values(["user_id", "session"]).reset_index(drop=True)
df["session_c"] = df["session"] - df["session"].mean()

# Create lagged variables
w_cols = [c[0] for c in CONSTRUCTS]
for col in w_cols:
    df[f"{col}_next"] = df.groupby("user_id")[col].shift(-1)

# Drop rows without next session
df_lag = df.dropna(subset=[f"{c}_next" for c in w_cols]).copy()

print(f"Data: {len(df)} total obs, {df['user_id'].nunique()} participants")
print(f"Lagged data: {len(df_lag)} obs (transitions)")
print()

# ── Run 5x5 CLPM ─────────────────────────────────────────────────────────
rows = []
for pred_col, _, pred_label in CONSTRUCTS:
    for out_col, _, out_label in CONSTRUCTS:
        outcome_next = f"{out_col}_next"
        formula = f"{outcome_next} ~ session_c + {pred_col} + {out_col} + C(user_id)"

        model = smf.ols(formula, data=df_lag).fit(
            cov_type="cluster", cov_kwds={"groups": df_lag["user_id"]}
        )

        beta = float(model.params[pred_col])
        se = float(model.bse[pred_col])
        t_val = float(model.tvalues[pred_col])
        p_val = float(model.pvalues[pred_col])

        rows.append({
            "predictor": pred_col,
            "predictor_label": pred_label,
            "outcome": out_col,
            "outcome_label": out_label,
            "beta": beta,
            "se": se,
            "t_value": t_val,
            "p_value": p_val,
            "n_obs": int(model.nobs),
            "r2": float(model.rsquared),
        })

clpm_df = pd.DataFrame(rows)

# Save CSV
csv_path = OUT_DIR / "analysis" / "crosslag_5x5_summary_v2.csv"
clpm_df.to_csv(csv_path, index=False)
print(f"Saved CLPM results to {csv_path}")

# Print significant cross-lagged paths
print("\n=== Significant Cross-Lagged Paths (p < .05) ===")
cross = clpm_df[clpm_df["predictor"] != clpm_df["outcome"]]
sig_paths = cross[cross["p_value"] < 0.05].sort_values("p_value")
for _, r in sig_paths.iterrows():
    stars = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*"
    print(f"  {r['predictor_label']:20s} → {r['outcome_label']:20s}  "
          f"β={r['beta']:.3f}  SE={r['se']:.3f}  p={r['p_value']:.4f} {stars}")

print("\n=== Autoregressive Stability ===")
auto = clpm_df[clpm_df["predictor"] == clpm_df["outcome"]]
for _, r in auto.iterrows():
    stars = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "n.s."
    print(f"  {r['predictor_label']:20s}  β={r['beta']:.3f}  p={r['p_value']:.4f} {stars}")

# ── Compare with old CSV ──────────────────────────────────────────────────
old_csv = Path(__file__).resolve().parent / "crosslag_5x5_summary_old.csv"
if old_csv.exists():
    old_df = pd.read_csv(old_csv)
    print("\n=== Comparison with old CLPM results ===")
    for _, new_row in sig_paths.iterrows():
        old_match = old_df[
            (old_df["predictor"] == new_row["predictor"]) &
            (old_df["outcome"] == new_row["outcome"])
        ]
        if len(old_match) > 0:
            old_row = old_match.iloc[0]
            diff = abs(new_row["beta"] - old_row["beta"])
            print(f"  {new_row['predictor_label']:20s} → {new_row['outcome_label']:20s}  "
                  f"old β={old_row['beta']:.3f}  new β={new_row['beta']:.3f}  "
                  f"diff={diff:.4f}  n_old={old_row['n_obs']} n_new={new_row['n_obs']}")

# ── Save results as JSON for paper reference ──────────────────────────────
results_json = {
    "model_specification": "outcome(t+1) ~ session_c + predictor(t) + outcome(t) + C(user_id)",
    "standard_errors": "cluster-robust (grouped by user_id)",
    "n_participants": int(df["user_id"].nunique()),
    "n_transitions": int(len(df_lag)),
    "significant_cross_lagged_paths": [],
    "autoregressive_coefficients": [],
}

for _, r in sig_paths.iterrows():
    results_json["significant_cross_lagged_paths"].append({
        "predictor": r["predictor_label"],
        "outcome": r["outcome_label"],
        "beta": round(r["beta"], 3),
        "se": round(r["se"], 3),
        "p": round(r["p_value"], 4),
    })

for _, r in auto.iterrows():
    results_json["autoregressive_coefficients"].append({
        "construct": r["predictor_label"],
        "beta": round(r["beta"], 3),
        "p": round(r["p_value"], 4),
    })

json_path = OUT_DIR / "analysis" / "clpm_results_v2.json"
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nSaved JSON results to {json_path}")


# ══════════════════════════════════════════════════════════════════════════
# PATH DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

cross_paths = clpm_df[clpm_df["predictor"] != clpm_df["outcome"]].copy()

constructs_layout = [
    ("w1_familiarity",   "Familiarity"),
    ("w2_penetration",   "Social\nPenetration"),
    ("w3_memory",        "Perceived\nMemory"),
    ("w4_quality",       "Conv.\nQuality"),
    ("w5_enjoy_intent",  "Enjoyment"),
]

# Layout: pentagon
n = len(constructs_layout)
angle_offset = np.pi / 2
angles = [angle_offset - 2 * np.pi * i / n for i in range(n)]
radius = 1.05
positions = {c[0]: (radius * np.cos(a), radius * np.sin(a))
             for c, a in zip(constructs_layout, angles)}

# Styling
SIG_COLOR = '#2E86AB'
NONSIG_COLOR = '#CCCCCC'
NODE_COLOR = '#F0F0F0'
NODE_EDGE = '#333333'

TITLE_SIZE = 8
NODE_SIZE = 6
ARROW_LABEL_SIZE = 5.5
LEGEND_SIZE = 6

fig, ax = plt.subplots(1, 1, figsize=(3.33, 3.3))

# Draw nodes
node_radius = 0.42
for key, label in constructs_layout:
    x, y = positions[key]
    circle = plt.Circle((x, y), node_radius, facecolor=NODE_COLOR,
                        edgecolor=NODE_EDGE, linewidth=1.0, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=NODE_SIZE, fontweight='bold', zorder=4)

# Collect all paths
all_paths = []
for _, row in cross_paths.iterrows():
    all_paths.append((row['predictor'], row['outcome'], row['beta'], row['p_value'], row['p_value'] < 0.05))

# Draw arrows
for pred, outcome, beta, p, sig in all_paths:
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

    # Offset for bidirectional arrows
    is_pen_mem = {pred, outcome} == {'w2_penetration', 'w3_memory'}
    perp_scale = 0.18 if is_pen_mem else 0.04
    arc_rad = 0.0 if is_pen_mem else 0.08
    perp_x, perp_y = -uy * perp_scale, ux * perp_scale
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
        lw = 0.6
        alpha = 0.4

    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', color=color,
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
    plt.Line2D([0], [0], color=NONSIG_COLOR, lw=0.6, alpha=0.5, label='n.s.'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=LEGEND_SIZE,
          frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
          bbox_to_anchor=(0.98, 0.02))

ax.set_title('Cross-Lagged Panel Paths ($t \\rightarrow t{+}1$)',
             fontsize=TITLE_SIZE, fontweight='bold', pad=5)

plt.tight_layout()

# Save
pdf_path = OUT_DIR / "figures" / "fig_path_diagram.pdf"
png_path = OUT_DIR / "figures" / "fig_path_diagram.png"
plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPath diagram saved to:\n  {pdf_path}\n  {png_path}")

"""Format growth curve statistics for paper inclusion."""

import json
from pathlib import Path

# Data from within_user_trends.csv
constructs = [
    {"id": "w1", "name": "Familiarity", "slope": 0.00487, "p": 0.879},
    {"id": "w2", "name": "Social Penetration", "slope": 0.08060, "p": 0.003},
    {"id": "w3", "name": "Perceived Memory", "slope": 0.00122, "p": 0.975},
    {"id": "w4", "name": "Conversational Quality", "slope": 0.04045, "p": 0.133},
    {"id": "w5", "name": "Enjoyment & Intent", "slope": 0.05733, "p": 0.069},
]

def sig_indicator(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "+"
    return "n.s."

# Text summary
lines = []
lines.append("Growth Curve Analysis: Linear Trends Across Sessions")
lines.append("=" * 55)
lines.append("")
lines.append("Mixed-effects linear growth models predicting each construct")
lines.append("from session number, with random intercepts and slopes per participant.")
lines.append("")
lines.append(f"{'Construct':<28} {'beta':>7} {'p':>8} {'Sig':>5}")
lines.append("-" * 55)

for c in constructs:
    sig = sig_indicator(c["p"])
    lines.append(f"{c['id']} {c['name']:<24} {c['slope']:>7.3f} {c['p']:>8.3f} {sig:>5}")

lines.append("-" * 55)
lines.append("")
lines.append("Significance: *** p<.001, ** p<.01, * p<.05, + p<.10, n.s. not significant")
lines.append("")
lines.append("Summary:")
lines.append("- Only w2 (Social Penetration) shows a significant positive linear")
lines.append("  trend across sessions (beta=0.081, p=.003), indicating that participants")
lines.append("  disclosed increasingly personal information over time.")
lines.append("- w5 (Enjoyment & Intent) shows a marginally significant positive trend")
lines.append("  (beta=0.057, p=.069).")
lines.append("- w1 (Familiarity), w3 (Perceived Memory), and w4 (Conversational Quality)")
lines.append("  do not show significant linear trends.")

txt_path = str(Path(__file__).resolve().parent / "growth_curve_summary.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# JSON version
json_data = {
    "analysis": "Linear growth curve models",
    "description": "Mixed-effects models predicting each construct from session number with random intercepts/slopes",
    "constructs": []
}

for c in constructs:
    json_data["constructs"].append({
        "id": c["id"],
        "name": c["name"],
        "beta_per_session": c["slope"],
        "p_value": c["p"],
        "significance": sig_indicator(c["p"]),
        "significant_at_05": c["p"] < 0.05,
    })

json_data["key_findings"] = [
    "Only Social Penetration (w2) shows significant growth (beta=0.081, p=.003)",
    "Enjoyment (w5) is marginally significant (beta=0.057, p=.069)",
    "Familiarity (w1), Perceived Memory (w3), and Conversational Quality (w4) are non-significant"
]

json_path = str(Path(__file__).resolve().parent / "growth_curve_summary.json")
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"\nSaved to:\n  {txt_path}\n  {json_path}")

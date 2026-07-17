"""
RI-CLPM (Random-Intercept Cross-Lagged Panel Model) analysis.

Approximates RI-CLPM by person-mean centering variables (removing between-person
variance) and running cross-lagged regressions on within-person deviations.

This separates within-person dynamics from between-person stable differences.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json
from pathlib import Path

# Load data
data_path = Path(__file__).resolve().parents[1] / "data" / "user_assessment_labels.csv"
df = pd.read_csv(data_path)

# Constructs: w2 = social_penetration (Q3,Q4), w3 = memory (Q5,Q6)
# Already computed in CSV as 'social_penetration' and 'memory'
df = df.rename(columns={"social_penetration": "w2", "memory": "w3"})

# Sort
df = df.sort_values(["user_id", "session"]).reset_index(drop=True)

# Step 1: Compute person means
person_means = df.groupby("user_id")[["w2", "w3"]].transform("mean")
df["w2_within"] = df["w2"] - person_means["w2"]
df["w3_within"] = df["w3"] - person_means["w3"]

# Step 2: Create lagged within-person variables (t+1)
df["w2_within_next"] = df.groupby("user_id")["w2_within"].shift(-1)
df["w3_within_next"] = df.groupby("user_id")["w3_within"].shift(-1)

# Drop rows without next session
df_lag = df.dropna(subset=["w2_within_next", "w3_within_next"]).copy()

print(f"N participants: {df['user_id'].nunique()}")
print(f"N observations (with lag): {len(df_lag)}")
print(f"Sessions per participant: {df_lag.groupby('user_id').size().describe()}")
print()

results = {}

# --- Path 1: w2(t) -> w3(t+1), controlling for w3(t) ---
# (Disclosure -> Memory, within-person)
model_w2_to_w3 = smf.ols(
    "w3_within_next ~ w2_within + w3_within",
    data=df_lag
).fit(cov_type="cluster", cov_kwds={"groups": df_lag["user_id"]})

print("=" * 60)
print("Within-person: w2 (Disclosure) at t -> w3 (Memory) at t+1")
print("Controlling for w3 at t (autoregressive)")
print("=" * 60)
print(model_w2_to_w3.summary2().tables[1])
print()

results["disclosure_to_memory"] = {
    "path": "w2(t) -> w3(t+1)",
    "description": "Within-person: Social Penetration predicting Perceived Memory",
    "beta_w2": float(model_w2_to_w3.params["w2_within"]),
    "se_w2": float(model_w2_to_w3.bse["w2_within"]),
    "t_w2": float(model_w2_to_w3.tvalues["w2_within"]),
    "p_w2": float(model_w2_to_w3.pvalues["w2_within"]),
    "beta_autoregressive": float(model_w2_to_w3.params["w3_within"]),
    "p_autoregressive": float(model_w2_to_w3.pvalues["w3_within"]),
    "r_squared": float(model_w2_to_w3.rsquared),
    "n_obs": int(model_w2_to_w3.nobs),
}

# --- Path 2: w3(t) -> w2(t+1), controlling for w2(t) ---
# (Memory -> Disclosure, within-person)
model_w3_to_w2 = smf.ols(
    "w2_within_next ~ w3_within + w2_within",
    data=df_lag
).fit(cov_type="cluster", cov_kwds={"groups": df_lag["user_id"]})

print("=" * 60)
print("Within-person: w3 (Memory) at t -> w2 (Disclosure) at t+1")
print("Controlling for w2 at t (autoregressive)")
print("=" * 60)
print(model_w3_to_w2.summary2().tables[1])
print()

results["memory_to_disclosure"] = {
    "path": "w3(t) -> w2(t+1)",
    "description": "Within-person: Perceived Memory predicting Social Penetration",
    "beta_w3": float(model_w3_to_w2.params["w3_within"]),
    "se_w3": float(model_w3_to_w2.bse["w3_within"]),
    "t_w3": float(model_w3_to_w2.tvalues["w3_within"]),
    "p_w3": float(model_w3_to_w2.pvalues["w3_within"]),
    "beta_autoregressive": float(model_w3_to_w2.params["w2_within"]),
    "p_autoregressive": float(model_w3_to_w2.pvalues["w2_within"]),
    "r_squared": float(model_w3_to_w2.rsquared),
    "n_obs": int(model_w3_to_w2.nobs),
}

# --- Summary comparison with traditional CLPM ---
print("=" * 60)
print("COMPARISON: Traditional CLPM vs RI-CLPM (within-person)")
print("=" * 60)
print(f"\nDisclosure -> Memory:")
print(f"  Traditional CLPM: beta=0.382, p<.001")
print(f"  RI-CLPM (within): beta={results['disclosure_to_memory']['beta_w2']:.3f}, "
      f"p={results['disclosure_to_memory']['p_w2']:.4f}")
print(f"\nMemory -> Disclosure:")
print(f"  Traditional CLPM: beta=0.157, p=.005")
print(f"  RI-CLPM (within): beta={results['memory_to_disclosure']['beta_w3']:.3f}, "
      f"p={results['memory_to_disclosure']['p_w3']:.4f}")

# Interpretation
d2m_sig = results["disclosure_to_memory"]["p_w2"] < 0.05
m2d_sig = results["memory_to_disclosure"]["p_w3"] < 0.05

interpretation = []
if d2m_sig:
    interpretation.append("Disclosure->Memory effect remains significant at within-person level.")
else:
    interpretation.append("Disclosure->Memory effect is non-significant at within-person level "
                          "(expected with N=24; direction consistency is key).")

if m2d_sig:
    interpretation.append("Memory->Disclosure effect remains significant at within-person level.")
else:
    interpretation.append("Memory->Disclosure effect is non-significant at within-person level "
                          "(expected with N=24; direction consistency is key).")

# Check direction consistency
d2m_same_dir = (results["disclosure_to_memory"]["beta_w2"] > 0)
m2d_same_dir = (results["memory_to_disclosure"]["beta_w3"] > 0)
interpretation.append(f"Direction consistency: Disclosure->Memory {'SAME' if d2m_same_dir else 'REVERSED'}, "
                      f"Memory->Disclosure {'SAME' if m2d_same_dir else 'REVERSED'}.")

results["interpretation"] = interpretation
results["n_participants"] = int(df["user_id"].nunique())
results["note"] = ("RI-CLPM approximated via person-mean centering. "
                   "With N=24, within-person effects are expected to be underpowered. "
                   "Direction consistency with the traditional CLPM is the primary validation criterion.")

print(f"\nInterpretation:")
for line in interpretation:
    print(f"  - {line}")

# Save
out_path = Path(__file__).resolve().parent / "riclpm_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

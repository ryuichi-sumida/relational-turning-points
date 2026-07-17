"""
Formal mediation analysis: Memory(t) → Disclosure(t+1) → Enjoyment(t+1)

Tests whether the effect of Perceived Memory on Enjoyment operates
indirectly through Self-Disclosure (Social Penetration).

Model:
  X = Perceived Memory at session t
  M = Social Penetration at session t+1
  Y = Enjoyment at session t+1

  Path a:  Memory(t)              → Social Penetration(t+1)  [controlling for SP(t)]
  Path b:  Social Penetration(t+1) → Enjoyment(t+1)          [controlling for Memory(t), Enjoyment(t)]
  Path c': Memory(t)              → Enjoyment(t+1)           [direct, controlling for SP(t+1), Enjoyment(t)]
  Indirect effect = a × b, with cluster bootstrap CI

All models include user fixed effects and cluster-robust SEs.
Bootstrap resamples at the participant level (cluster bootstrap).
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "user_assessment_labels.csv"
N_BOOT = 5000
SEED = 42
ALPHA = 0.05

# ── Load and prepare data ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Rename to match paper notation
df = df.rename(columns={
    "social_penetration": "w2_sp",
    "memory": "w3_mem",
    "enjoyment": "w5_enj",
    "familiarity": "w1_fam",
    "conversational": "w4_cq",
})

df = df.sort_values(["user_id", "session"]).reset_index(drop=True)

# Center session
df["session_c"] = df["session"] - df["session"].mean()

# Create lagged (next-session) variables
for col in ["w2_sp", "w3_mem", "w5_enj", "w4_cq"]:
    df[f"{col}_next"] = df.groupby("user_id")[col].shift(-1)

# Drop rows without a next session
df_lag = df.dropna(subset=["w2_sp_next", "w5_enj_next"]).copy()

print(f"Data: {len(df)} total obs, {df['user_id'].nunique()} participants")
print(f"Lagged data: {len(df_lag)} obs after creating t+1 variables")
print()


# ── Helper: fit OLS with user FE and cluster-robust SE ─────────────────
def fit_fe_model(formula, data):
    """Fit OLS with user fixed effects and cluster-robust SEs."""
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["user_id"]}
    )
    return model


# ── Path a: Memory(t) → Social Penetration(t+1) ──────────────────────
print("=" * 70)
print("PATH a: Memory(t) → Social Penetration(t+1)")
print("  Controls: SP(t), session, user FE")
print("=" * 70)

formula_a = "w2_sp_next ~ w3_mem + w2_sp + session_c + C(user_id)"
model_a = fit_fe_model(formula_a, df_lag)

a_coef = model_a.params["w3_mem"]
a_se = model_a.bse["w3_mem"]
a_p = model_a.pvalues["w3_mem"]
print(f"  a = {a_coef:.4f}, SE = {a_se:.4f}, p = {a_p:.4f}")
print()


# ── Path b + c': Enjoyment(t+1) ~ SP(t+1) + Memory(t) ────────────────
print("=" * 70)
print("PATH b + c': Enjoyment(t+1) ~ SP(t+1) + Memory(t)")
print("  Controls: Enjoyment(t), session, user FE")
print("=" * 70)

formula_bc = "w5_enj_next ~ w2_sp_next + w3_mem + w5_enj + session_c + C(user_id)"
model_bc = fit_fe_model(formula_bc, df_lag)

b_coef = model_bc.params["w2_sp_next"]
b_se = model_bc.bse["w2_sp_next"]
b_p = model_bc.pvalues["w2_sp_next"]

cprime_coef = model_bc.params["w3_mem"]
cprime_se = model_bc.bse["w3_mem"]
cprime_p = model_bc.pvalues["w3_mem"]

print(f"  b  (SP(t+1) → Enj(t+1))  = {b_coef:.4f}, SE = {b_se:.4f}, p = {b_p:.4f}")
print(f"  c' (Mem(t)  → Enj(t+1))  = {cprime_coef:.4f}, SE = {cprime_se:.4f}, p = {cprime_p:.4f}")
print()


# ── Path c (total): Memory(t) → Enjoyment(t+1) without mediator ──────
print("=" * 70)
print("PATH c (total): Memory(t) → Enjoyment(t+1)")
print("  Controls: Enjoyment(t), session, user FE")
print("=" * 70)

formula_c = "w5_enj_next ~ w3_mem + w5_enj + session_c + C(user_id)"
model_c = fit_fe_model(formula_c, df_lag)

c_coef = model_c.params["w3_mem"]
c_se = model_c.bse["w3_mem"]
c_p = model_c.pvalues["w3_mem"]
print(f"  c = {c_coef:.4f}, SE = {c_se:.4f}, p = {c_p:.4f}")
print()


# ── Indirect effect: a × b ────────────────────────────────────────────
indirect = a_coef * b_coef
print("=" * 70)
print("INDIRECT EFFECT: a × b")
print("=" * 70)
print(f"  Indirect = {a_coef:.4f} × {b_coef:.4f} = {indirect:.4f}")
print()


# ── Cluster bootstrap for indirect effect CI ──────────────────────────
print("=" * 70)
print(f"CLUSTER BOOTSTRAP ({N_BOOT} iterations)")
print("=" * 70)

rng = np.random.default_rng(SEED)
user_ids = df_lag["user_id"].unique()
n_users = len(user_ids)

boot_a = np.zeros(N_BOOT)
boot_b = np.zeros(N_BOOT)
boot_indirect = np.zeros(N_BOOT)
boot_cprime = np.zeros(N_BOOT)
n_failed = 0

for i in range(N_BOOT):
    if (i + 1) % 1000 == 0:
        print(f"  ... bootstrap iteration {i + 1}/{N_BOOT}")

    # Resample participants (cluster bootstrap)
    sampled_ids = rng.choice(user_ids, size=n_users, replace=True)

    # Build bootstrap sample (preserving within-user structure)
    frames = []
    for idx, uid in enumerate(sampled_ids):
        sub = df_lag[df_lag["user_id"] == uid].copy()
        sub["boot_uid"] = idx  # unique ID for this bootstrap copy
        frames.append(sub)
    boot_df = pd.concat(frames, ignore_index=True)

    try:
        # Path a
        m_a = smf.ols(
            "w2_sp_next ~ w3_mem + w2_sp + session_c + C(boot_uid)",
            data=boot_df,
        ).fit()
        boot_a[i] = m_a.params["w3_mem"]

        # Path b + c'
        m_bc = smf.ols(
            "w5_enj_next ~ w2_sp_next + w3_mem + w5_enj + session_c + C(boot_uid)",
            data=boot_df,
        ).fit()
        boot_b[i] = m_bc.params["w2_sp_next"]
        boot_cprime[i] = m_bc.params["w3_mem"]

        boot_indirect[i] = boot_a[i] * boot_b[i]
    except Exception:
        boot_indirect[i] = np.nan
        n_failed += 1

# Remove failed iterations
boot_indirect_clean = boot_indirect[~np.isnan(boot_indirect)]
boot_a_clean = boot_a[~np.isnan(boot_indirect)]
boot_b_clean = boot_b[~np.isnan(boot_indirect)]
boot_cprime_clean = boot_cprime[~np.isnan(boot_indirect)]

print(f"  Completed: {len(boot_indirect_clean)}/{N_BOOT} successful iterations")
print()

# Percentile CI
ci_lo = np.percentile(boot_indirect_clean, 100 * ALPHA / 2)
ci_hi = np.percentile(boot_indirect_clean, 100 * (1 - ALPHA / 2))

# Bias-corrected accelerated (BCa) CI
z0 = np.quantile(
    (boot_indirect_clean < indirect).astype(float),
    1.0,  # proportion below point estimate
)
# Simpler: use percentile method which is standard for mediation
# BCa is complex; percentile + bias-corrected percentile are most common

# Bias-corrected CI
prop_below = np.mean(boot_indirect_clean < indirect)
from scipy.stats import norm

z0 = norm.ppf(prop_below) if 0 < prop_below < 1 else 0.0
z_alpha_lo = norm.ppf(ALPHA / 2)
z_alpha_hi = norm.ppf(1 - ALPHA / 2)

bc_lo_pct = norm.cdf(2 * z0 + z_alpha_lo) * 100
bc_hi_pct = norm.cdf(2 * z0 + z_alpha_hi) * 100
bc_lo = np.percentile(boot_indirect_clean, bc_lo_pct)
bc_hi = np.percentile(boot_indirect_clean, bc_hi_pct)

# Bootstrap p-value (proportion of bootstrap indirect effects ≤ 0)
boot_p = np.mean(boot_indirect_clean <= 0)
# Two-tailed p-value
boot_p_two = 2 * min(boot_p, 1 - boot_p)


# ── Summary ───────────────────────────────────────────────────────────
print("=" * 70)
print("MEDIATION ANALYSIS SUMMARY")
print("  X = Perceived Memory(t)")
print("  M = Social Penetration(t+1)")
print("  Y = Enjoyment(t+1)")
print("=" * 70)
print()
print(f"  Path a  (X → M):   β = {a_coef:.4f},  p = {a_p:.4f}")
print(f"  Path b  (M → Y):   β = {b_coef:.4f},  p = {b_p:.4f}")
print(f"  Path c' (X → Y):   β = {cprime_coef:.4f},  p = {cprime_p:.4f}  [direct]")
print(f"  Path c  (X → Y):   β = {c_coef:.4f},  p = {c_p:.4f}  [total]")
print()
print(f"  Indirect (a×b):    {indirect:.4f}")
print(f"  Percentile 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Bias-corrected CI: [{bc_lo:.4f}, {bc_hi:.4f}]")
print(f"  Bootstrap p:       {boot_p_two:.4f}")
print()

if ci_lo > 0 or ci_hi < 0:
    print("  ✓ Indirect effect is SIGNIFICANT (95% CI excludes zero)")
else:
    print("  ✗ Indirect effect is NOT significant (95% CI includes zero)")

if bc_lo > 0 or bc_hi < 0:
    print("  ✓ Indirect effect is SIGNIFICANT (bias-corrected 95% CI excludes zero)")
else:
    print("  ✗ Indirect effect is NOT significant (bias-corrected 95% CI includes zero)")

# Proportion mediated
if abs(c_coef) > 1e-8:
    prop_mediated = indirect / c_coef
    print(f"\n  Proportion mediated: {prop_mediated:.1%}")

print()
print("─" * 70)
print("Boot distribution of indirect effect:")
print(f"  Mean:   {np.mean(boot_indirect_clean):.4f}")
print(f"  Median: {np.median(boot_indirect_clean):.4f}")
print(f"  SD:     {np.std(boot_indirect_clean):.4f}")
print(f"  2.5%:   {np.percentile(boot_indirect_clean, 2.5):.4f}")
print(f"  97.5%:  {np.percentile(boot_indirect_clean, 97.5):.4f}")
print()

# ── Also run: concurrent mediation (same-session) for comparison ──────
print()
print("=" * 70)
print("SUPPLEMENTARY: CONCURRENT MEDIATION (same session)")
print("  X = Memory(t),  M = SP(t),  Y = Enjoyment(t)")
print("=" * 70)
print()

# Path a_c: Memory(t) → SP(t) [concurrent]
formula_a_conc = "w2_sp ~ w3_mem + session_c + C(user_id)"
model_a_conc = fit_fe_model(formula_a_conc, df)
a_conc = model_a_conc.params["w3_mem"]
a_conc_p = model_a_conc.pvalues["w3_mem"]

# Path b_c + c'_c: Enjoyment(t) ~ SP(t) + Memory(t)
formula_bc_conc = "w5_enj ~ w2_sp + w3_mem + session_c + C(user_id)"
model_bc_conc = fit_fe_model(formula_bc_conc, df)
b_conc = model_bc_conc.params["w2_sp"]
b_conc_p = model_bc_conc.pvalues["w2_sp"]
cprime_conc = model_bc_conc.params["w3_mem"]
cprime_conc_p = model_bc_conc.pvalues["w3_mem"]

# Total effect
formula_c_conc = "w5_enj ~ w3_mem + session_c + C(user_id)"
model_c_conc = fit_fe_model(formula_c_conc, df)
c_conc = model_c_conc.params["w3_mem"]
c_conc_p = model_c_conc.pvalues["w3_mem"]

indirect_conc = a_conc * b_conc

# Bootstrap for concurrent
boot_indirect_conc = np.zeros(N_BOOT)
for i in range(N_BOOT):
    sampled_ids = rng.choice(user_ids, size=n_users, replace=True)
    frames = []
    for idx, uid in enumerate(sampled_ids):
        sub = df[df["user_id"] == uid].copy()
        sub["boot_uid"] = idx
        frames.append(sub)
    boot_df = pd.concat(frames, ignore_index=True)
    try:
        m_a_c = smf.ols("w2_sp ~ w3_mem + session_c + C(boot_uid)", data=boot_df).fit()
        m_bc_c = smf.ols("w5_enj ~ w2_sp + w3_mem + session_c + C(boot_uid)", data=boot_df).fit()
        boot_indirect_conc[i] = m_a_c.params["w3_mem"] * m_bc_c.params["w2_sp"]
    except Exception:
        boot_indirect_conc[i] = np.nan

boot_ind_conc_clean = boot_indirect_conc[~np.isnan(boot_indirect_conc)]
ci_conc_lo = np.percentile(boot_ind_conc_clean, 2.5)
ci_conc_hi = np.percentile(boot_ind_conc_clean, 97.5)

print(f"  Path a  (Mem → SP):   β = {a_conc:.4f},  p = {a_conc_p:.4f}")
print(f"  Path b  (SP → Enj):   β = {b_conc:.4f},  p = {b_conc_p:.4f}")
print(f"  Path c' (Mem → Enj):  β = {cprime_conc:.4f},  p = {cprime_conc_p:.4f}  [direct]")
print(f"  Path c  (Mem → Enj):  β = {c_conc:.4f},  p = {c_conc_p:.4f}  [total]")
print(f"  Indirect (a×b):       {indirect_conc:.4f}")
print(f"  Percentile 95% CI:    [{ci_conc_lo:.4f}, {ci_conc_hi:.4f}]")
if ci_conc_lo > 0 or ci_conc_hi < 0:
    print("  ✓ Concurrent indirect effect is SIGNIFICANT")
else:
    print("  ✗ Concurrent indirect effect is NOT significant")
print()

# Also include Conversational Quality as parallel mediator
print("=" * 70)
print("SUPPLEMENTARY: PARALLEL MEDIATION (SP + CQ → Enjoyment)")
print("  X = Memory(t),  M1 = SP(t+1),  M2 = CQ(t+1),  Y = Enjoyment(t+1)")
print("=" * 70)
print()

# Path a1: Memory(t) → SP(t+1)
# Already computed as model_a above
a1 = a_coef
a1_p = a_p

# Path a2: Memory(t) → CQ(t+1)
formula_a2 = "w4_cq_next ~ w3_mem + w4_cq + session_c + C(user_id)"
model_a2 = fit_fe_model(formula_a2, df_lag)
a2 = model_a2.params["w3_mem"]
a2_p = model_a2.pvalues["w3_mem"]

# Paths b1, b2, c': Enjoyment(t+1) ~ SP(t+1) + CQ(t+1) + Memory(t)
formula_parallel = "w5_enj_next ~ w2_sp_next + w4_cq_next + w3_mem + w5_enj + session_c + C(user_id)"
model_parallel = fit_fe_model(formula_parallel, df_lag)
b1_par = model_parallel.params["w2_sp_next"]
b1_par_p = model_parallel.pvalues["w2_sp_next"]
b2_par = model_parallel.params["w4_cq_next"]
b2_par_p = model_parallel.pvalues["w4_cq_next"]
cprime_par = model_parallel.params["w3_mem"]
cprime_par_p = model_parallel.pvalues["w3_mem"]

indirect1 = a1 * b1_par
indirect2 = a2 * b2_par

print(f"  Path a1 (Mem → SP):     β = {a1:.4f},  p = {a1_p:.4f}")
print(f"  Path a2 (Mem → CQ):     β = {a2:.4f},  p = {a2_p:.4f}")
print(f"  Path b1 (SP → Enj):     β = {b1_par:.4f},  p = {b1_par_p:.4f}")
print(f"  Path b2 (CQ → Enj):     β = {b2_par:.4f},  p = {b2_par_p:.4f}")
print(f"  Path c' (Mem → Enj):    β = {cprime_par:.4f},  p = {cprime_par_p:.4f}  [direct]")
print(f"  Indirect via SP (a1×b1):  {indirect1:.4f}")
print(f"  Indirect via CQ (a2×b2):  {indirect2:.4f}")
print(f"  Total indirect:           {indirect1 + indirect2:.4f}")
print()

# Bootstrap for parallel mediation
boot_ind1 = np.zeros(N_BOOT)
boot_ind2 = np.zeros(N_BOOT)
for i in range(N_BOOT):
    sampled_ids = rng.choice(user_ids, size=n_users, replace=True)
    frames = []
    for idx, uid in enumerate(sampled_ids):
        sub = df_lag[df_lag["user_id"] == uid].copy()
        sub["boot_uid"] = idx
        frames.append(sub)
    boot_df = pd.concat(frames, ignore_index=True)
    try:
        m_a1 = smf.ols("w2_sp_next ~ w3_mem + w2_sp + session_c + C(boot_uid)", data=boot_df).fit()
        m_a2 = smf.ols("w4_cq_next ~ w3_mem + w4_cq + session_c + C(boot_uid)", data=boot_df).fit()
        m_par = smf.ols("w5_enj_next ~ w2_sp_next + w4_cq_next + w3_mem + w5_enj + session_c + C(boot_uid)", data=boot_df).fit()
        boot_ind1[i] = m_a1.params["w3_mem"] * m_par.params["w2_sp_next"]
        boot_ind2[i] = m_a2.params["w3_mem"] * m_par.params["w4_cq_next"]
    except Exception:
        boot_ind1[i] = np.nan
        boot_ind2[i] = np.nan

mask = ~np.isnan(boot_ind1)
boot_ind1_clean = boot_ind1[mask]
boot_ind2_clean = boot_ind2[mask]

ci_ind1 = (np.percentile(boot_ind1_clean, 2.5), np.percentile(boot_ind1_clean, 97.5))
ci_ind2 = (np.percentile(boot_ind2_clean, 2.5), np.percentile(boot_ind2_clean, 97.5))

print(f"  Indirect via SP:  {indirect1:.4f}  95% CI [{ci_ind1[0]:.4f}, {ci_ind1[1]:.4f}]", end="")
print("  ✓ SIG" if ci_ind1[0] > 0 or ci_ind1[1] < 0 else "  ✗ n.s.")
print(f"  Indirect via CQ:  {indirect2:.4f}  95% CI [{ci_ind2[0]:.4f}, {ci_ind2[1]:.4f}]", end="")
print("  ✓ SIG" if ci_ind2[0] > 0 or ci_ind2[1] < 0 else "  ✗ n.s.")
print()

"""
CFA and Discriminant Validity Analysis
=======================================
Tests the 5-factor structure of the 10-item relational quality questionnaire.
Factor structure:
  Familiarity:        Q1, Q2
  Social Penetration: Q3, Q4
  Memory:             Q5, Q6
  Conversational:     Q7, Q8
  Enjoyment:          Q9, Q10
"""

import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import semopy
from itertools import combinations
from scipy import stats

warnings.filterwarnings("ignore")

DATA_PATH = str(Path(__file__).resolve().parents[1] / "data" / "user_assessment_labels.csv")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "cfa_results.json")

# Factor definitions
FACTORS = {
    "Familiarity": ["Q1", "Q2"],
    "SocPen": ["Q3", "Q4"],
    "Memory": ["Q5", "Q6"],
    "ConvQual": ["Q7", "Q8"],
    "Enjoyment": ["Q9", "Q10"],
}

FACTOR_NAMES = list(FACTORS.keys())
ALL_ITEMS = [item for items in FACTORS.values() for item in items]


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Data: {len(df)} observations, {df['user_id'].nunique()} participants")
    print(f"Sessions per participant: {df.groupby('user_id')['session'].count().describe()[['min','max','mean']].to_dict()}")
    return df


def build_5factor_model():
    lines = []
    for factor, items in FACTORS.items():
        lines.append(f"{factor} =~ {' + '.join(items)}")
    return "\n".join(lines)


def build_1factor_model():
    return f"General =~ {' + '.join(ALL_ITEMS)}"


def compute_srmr(model, data):
    """Compute SRMR (Standardized Root Mean Square Residual)."""
    S = data.corr().values  # observed correlation matrix
    n_vars = S.shape[0]
    # Get model-implied covariance matrix and convert to correlation
    sigma = model.calc_sigma()[0]
    # Convert to correlation
    d = np.sqrt(np.diag(sigma))
    sigma_corr = sigma / np.outer(d, d)
    # SRMR = sqrt(mean of squared residual correlations in lower triangle)
    residuals = S - sigma_corr
    lower_tri = np.tril_indices(n_vars, k=-1)
    # Include diagonal too (should be ~0)
    srmr = np.sqrt(np.mean(residuals[lower_tri] ** 2))
    return float(srmr)


def compute_rmsea_ci(chi2, df_model, n, alpha=0.10):
    """Compute RMSEA and its 90% CI using the non-central chi-square approach."""
    if df_model <= 0 or n <= 0:
        return None, None
    # Point estimate
    rmsea = np.sqrt(max((chi2 - df_model) / (df_model * (n - 1)), 0))

    # 90% CI via non-central chi-square
    from scipy.optimize import brentq

    # Lower bound
    def lower_func(ncp):
        return stats.ncx2.cdf(chi2, df_model, ncp) - (1 - alpha / 2)
    try:
        ncp_lower = brentq(lower_func, 0, max(chi2 * 5, 100))
        rmsea_lower = np.sqrt(ncp_lower / (df_model * (n - 1)))
    except (ValueError, RuntimeError):
        rmsea_lower = 0.0

    # Upper bound
    def upper_func(ncp):
        return stats.ncx2.cdf(chi2, df_model, ncp) - (alpha / 2)
    try:
        ncp_upper = brentq(upper_func, 0, max(chi2 * 5, 100))
        rmsea_upper = np.sqrt(ncp_upper / (df_model * (n - 1)))
    except (ValueError, RuntimeError):
        rmsea_upper = None

    return rmsea_lower, rmsea_upper


def extract_fit_indices(model, data):
    """Extract fit indices from a fitted semopy model."""
    stats_df = semopy.calc_stats(model)
    result = {}
    for col in stats_df.columns:
        val = stats_df[col].values[0]
        if isinstance(val, (np.floating, float)):
            result[col] = float(val)
        elif isinstance(val, (np.integer, int)):
            result[col] = int(val)
        else:
            result[col] = str(val)

    # Add SRMR
    result["SRMR"] = compute_srmr(model, data)

    # Add RMSEA 90% CI
    chi2 = result.get("chi2")
    dof = result.get("DoF")
    n = len(data)
    if chi2 is not None and dof is not None:
        lo, hi = compute_rmsea_ci(float(chi2), int(dof), n)
        result["RMSEA_CI_lower"] = lo
        result["RMSEA_CI_upper"] = hi

    return result


def run_cfa(df):
    """Run 5-factor and 1-factor CFA models."""
    data = df[ALL_ITEMS].copy()

    # Model 1: 5-factor
    spec_5f = build_5factor_model()
    print("\n=== 5-Factor CFA Model ===")
    print(spec_5f)
    model_5f = semopy.Model(spec_5f)
    model_5f.fit(data)
    fit_5f = extract_fit_indices(model_5f, data)
    print("\nFit indices (5-factor):")
    for k, v in fit_5f.items():
        print(f"  {k}: {v}")

    # Model 2: 1-factor
    spec_1f = build_1factor_model()
    print("\n=== 1-Factor CFA Model ===")
    print(spec_1f)
    model_1f = semopy.Model(spec_1f)
    model_1f.fit(data)
    fit_1f = extract_fit_indices(model_1f, data)
    print("\nFit indices (1-factor):")
    for k, v in fit_1f.items():
        print(f"  {k}: {v}")

    # Chi-square difference test
    chi2_5f = fit_5f.get("chi2", None)
    chi2_1f = fit_1f.get("chi2", None)
    df_5f = fit_5f.get("DoF", None)
    df_1f = fit_1f.get("DoF", None)

    diff_test = {}
    if chi2_5f is not None and chi2_1f is not None and df_5f is not None and df_1f is not None:
        delta_chi2 = float(chi2_1f) - float(chi2_5f)
        delta_df = int(df_1f) - int(df_5f)
        p_diff = float(1 - stats.chi2.cdf(delta_chi2, delta_df)) if delta_df > 0 else None
        diff_test = {
            "delta_chi2": float(delta_chi2),
            "delta_df": int(delta_df),
            "p_value": p_diff,
        }
        print(f"\n=== Chi-Square Difference Test ===")
        if p_diff is not None:
            print(f"  Delta chi2 = {delta_chi2:.3f}, Delta df = {delta_df}, p = {p_diff:.6f}")
        else:
            print("  Could not compute")

    # Extract standardized loadings from 5-factor model
    # semopy inspect: loadings have op="~", lval=item, rval=factor
    inspect_df = model_5f.inspect(std_est=True)
    loadings = inspect_df[inspect_df["op"] == "~"].copy()
    print("\n=== Standardized Factor Loadings (5-factor) ===")
    print(loadings[["lval", "rval", "Estimate", "Est. Std"]].to_string(index=False))

    # Build loadings_dict keyed by factor name
    loadings_dict = {}
    for _, row in loadings.iterrows():
        item = row["lval"]    # item is on the left in semopy
        factor = row["rval"]  # factor is on the right
        loadings_dict.setdefault(factor, {})[item] = {
            "unstd": float(row["Estimate"]),
            "std": float(row["Est. Std"]),
        }

    return fit_5f, fit_1f, diff_test, loadings_dict, model_5f


def compute_ave(loadings_dict):
    """Compute Average Variance Extracted per factor from standardized loadings."""
    ave = {}
    for factor, items in loadings_dict.items():
        std_loadings = [v["std"] for v in items.values()]
        ave[factor] = float(np.mean([l**2 for l in std_loadings]))
    return ave


def compute_htmt(df):
    """
    Compute HTMT ratios for all pairs of constructs.
    HTMT = mean(between-construct correlations) / geometric_mean(within-construct mean correlations)
    """
    data = df[ALL_ITEMS]
    corr = data.corr()

    htmt_matrix = {}
    factor_pairs = list(combinations(FACTOR_NAMES, 2))

    for f1, f2 in factor_pairs:
        items_f1 = FACTORS[f1]
        items_f2 = FACTORS[f2]

        # Between-construct correlations (absolute)
        between = []
        for i1 in items_f1:
            for i2 in items_f2:
                between.append(abs(corr.loc[i1, i2]))
        mean_between = np.mean(between)

        # Within-construct correlations for f1
        within_f1 = []
        for i, j in combinations(items_f1, 2):
            within_f1.append(abs(corr.loc[i, j]))
        mean_within_f1 = np.mean(within_f1) if within_f1 else 1.0

        # Within-construct correlations for f2
        within_f2 = []
        for i, j in combinations(items_f2, 2):
            within_f2.append(abs(corr.loc[i, j]))
        mean_within_f2 = np.mean(within_f2) if within_f2 else 1.0

        # Geometric mean of within-construct correlations
        geo_mean_within = np.sqrt(mean_within_f1 * mean_within_f2)

        htmt_val = mean_between / geo_mean_within if geo_mean_within > 0 else np.nan
        htmt_matrix[f"{f1}-{f2}"] = float(htmt_val)

    return htmt_matrix


def compute_correlations(df, level="session"):
    """Compute inter-construct correlations."""
    # CSV columns are lowercase
    cols = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
    if level == "person_mean":
        agg = df.groupby("user_id")[cols].mean()
        corr = agg.corr()
    else:
        corr = df[cols].corr()

    # Rename for clarity
    corr.index = FACTOR_NAMES
    corr.columns = FACTOR_NAMES
    return corr


def fornell_larcker(ave, corr_matrix):
    """Check Fornell-Larcker criterion: sqrt(AVE) > inter-construct correlations."""
    results = {}
    sqrt_ave = {f: np.sqrt(v) for f, v in ave.items()}

    for f in FACTOR_NAMES:
        results[f] = {
            "AVE": ave[f],
            "sqrt_AVE": sqrt_ave[f],
            "max_corr_with_other": 0.0,
            "passes": True,
        }
        for f2 in FACTOR_NAMES:
            if f != f2:
                r = abs(corr_matrix.loc[f, f2])
                if r > results[f]["max_corr_with_other"]:
                    results[f]["max_corr_with_other"] = float(r)
        results[f]["passes"] = bool(sqrt_ave[f] > results[f]["max_corr_with_other"])

    return results


def main():
    df = load_data()

    # (A) CFA
    fit_5f, fit_1f, diff_test, loadings_dict, model_5f = run_cfa(df)

    # (D) AVE
    ave = compute_ave(loadings_dict)
    print("\n=== Average Variance Extracted (AVE) ===")
    for f, v in ave.items():
        status = "PASS" if v > 0.50 else "FAIL"
        print(f"  {f}: {v:.3f} [{status}]")

    # (B) HTMT
    htmt_session = compute_htmt(df)
    print("\n=== HTMT Ratios (Session Level) ===")
    for pair, val in htmt_session.items():
        status = "PASS" if val < 0.85 else ("MARGINAL" if val < 0.90 else "FAIL")
        print(f"  {pair}: {val:.3f} [{status}]")

    # Also compute HTMT on person-mean data
    person_means = df.groupby("user_id")[ALL_ITEMS].mean().reset_index()
    htmt_person = compute_htmt(person_means)
    print("\n=== HTMT Ratios (Person-Mean Level) ===")
    for pair, val in htmt_person.items():
        status = "PASS" if val < 0.85 else ("MARGINAL" if val < 0.90 else "FAIL")
        print(f"  {pair}: {val:.3f} [{status}]")

    # (C) Inter-construct correlations
    corr_session = compute_correlations(df, level="session")
    corr_person = compute_correlations(df, level="person_mean")
    print("\n=== Inter-Construct Correlations (Session Level) ===")
    print(corr_session.round(3).to_string())
    print("\n=== Inter-Construct Correlations (Person-Mean Level) ===")
    print(corr_person.round(3).to_string())

    # (D) Fornell-Larcker
    fl_session = fornell_larcker(ave, corr_session)
    print("\n=== Fornell-Larcker Criterion (Session Level) ===")
    for f, info in fl_session.items():
        status = "PASS" if info["passes"] else "FAIL"
        print(f"  {f}: sqrt(AVE)={info['sqrt_AVE']:.3f} > max_r={info['max_corr_with_other']:.3f} => {status}")

    fl_person = fornell_larcker(ave, corr_person)
    print("\n=== Fornell-Larcker Criterion (Person-Mean Level) ===")
    for f, info in fl_person.items():
        status = "PASS" if info["passes"] else "FAIL"
        print(f"  {f}: sqrt(AVE)={info['sqrt_AVE']:.3f} > max_r={info['max_corr_with_other']:.3f} => {status}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\n{'Model':<12} {'chi2':>10} {'df':>5} {'p':>10} {'CFI':>7} {'RMSEA':>7} {'RMSEA 90% CI':>16} {'SRMR':>7}")
    print("-" * 90)
    for name, fit in [("5-Factor", fit_5f), ("1-Factor", fit_1f)]:
        chi2 = fit.get("chi2", "N/A")
        dof = fit.get("DoF", "N/A")
        p = fit.get("chi2 p-value", "N/A")
        cfi = fit.get("CFI", "N/A")
        rmsea = fit.get("RMSEA", "N/A")
        rmsea_lo = fit.get("RMSEA_CI_lower", None)
        rmsea_hi = fit.get("RMSEA_CI_upper", None)
        srmr = fit.get("SRMR", "N/A")
        chi2_s = f"{chi2:.3f}" if isinstance(chi2, float) else str(chi2)
        p_s = f"{p:.4f}" if isinstance(p, float) else str(p)
        cfi_s = f"{cfi:.3f}" if isinstance(cfi, float) else str(cfi)
        rmsea_s = f"{rmsea:.3f}" if isinstance(rmsea, float) else str(rmsea)
        if rmsea_lo is not None and rmsea_hi is not None:
            rmsea_ci_s = f"[{rmsea_lo:.3f}, {rmsea_hi:.3f}]"
        else:
            rmsea_ci_s = "N/A"
        srmr_s = f"{srmr:.3f}" if isinstance(srmr, float) else str(srmr)
        print(f"{name:<12} {chi2_s:>10} {dof:>5} {p_s:>10} {cfi_s:>7} {rmsea_s:>7} {rmsea_ci_s:>16} {srmr_s:>7}")

    if diff_test:
        print(f"\nChi-square difference: Delta_chi2={diff_test['delta_chi2']:.3f}, Delta_df={diff_test['delta_df']}, p={diff_test['p_value']:.6f}")

    # Save results
    results = {
        "n_observations": len(df),
        "n_participants": int(df["user_id"].nunique()),
        "cfa_5factor": fit_5f,
        "cfa_1factor": fit_1f,
        "chi2_difference_test": diff_test,
        "standardized_loadings": loadings_dict,
        "ave": ave,
        "htmt_session_level": htmt_session,
        "htmt_person_mean_level": htmt_person,
        "inter_construct_correlations_session": corr_session.round(4).to_dict(),
        "inter_construct_correlations_person_mean": corr_person.round(4).to_dict(),
        "fornell_larcker_session": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()} for k, v in fl_session.items()},
        "fornell_larcker_person_mean": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()} for k, v in fl_person.items()},
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

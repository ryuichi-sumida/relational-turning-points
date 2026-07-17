"""
Compare expert annotations with user self-reported ratings.
Computes MAE, Pearson correlation, and Concordance Correlation Coefficient (CCC).
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Corrupted video files/annotations to skip
BAD_VIDEOS = {
    (21, 9),
    (21, 10),
    (25, 6),
    (25, 10),
    (38, 9),
    (45, 5),
    (45, 8),
    (47, 1),
    (50, 6),
}
BAD_ID = {29, 30}


def parse_annotation_file(filepath):
    """Parse the annotation file to extract expert ratings."""
    df = pd.read_excel(filepath, sheet_name='annotation', header=None)

    annotations = []
    current_user_id = None

    for idx, row in df.iterrows():
        # Check if this is a user ID row (numeric value in column 0, header text in column 1)
        if pd.notna(row[0]) and str(row[1]).startswith('Q1:'):
            current_user_id = int(row[0])
            continue

        # Check if this is a session row
        if pd.notna(row[0]) and str(row[0]).startswith('session'):
            session_num = int(str(row[0]).replace('session', ''))

            # Extract Q1-Q10 ratings (columns 1-10)
            ratings = {}
            for q in range(1, 11):
                val = row[q]
                if pd.notna(val):
                    try:
                        ratings[f'Q{q}'] = float(val)
                    except:
                        ratings[f'Q{q}'] = np.nan
                else:
                    ratings[f'Q{q}'] = np.nan

            annotations.append({
                'user_id': current_user_id,
                'session': session_num,
                **ratings
            })

    return pd.DataFrame(annotations)


def parse_user_assessment_file(filepath):
    """Parse the user assessment file to extract user self-reported ratings."""
    df = pd.read_excel(filepath)

    # Rename columns for easier access
    col_mapping = {
        'メールに記載されているユーザーIDを答えてください（e.g., 01, 12など)': 'user_id',
        'これは何回目の会話ですか？（1~10回のうちどれですか？）': 'session',
        'Q1: この会話AIに対して親しみを感じる。': 'Q1',
        'Q2: この会話AIは私の気持ちに寄り添ってくれると感じる。': 'Q2',
        'Q3: この会話AIとは、自分の生活のさまざまな領域に関わる幅広い話題についても安心して話すことができる。': 'Q3',
        'Q4: この会話AIには、悩みや気持ちなど、より深く個人的なことについても安心して話すことができる。': 'Q4',
        'Q5: この会話AIは過去の会話の内容を覚えてくれていると感じる。': 'Q5',
        'Q6: この会話AIは私のことを理解してくれていると感じる。': 'Q6',
        'Q7: この会話AIとの会話は自然に感じる。': 'Q7',
        'Q8: この会話AIの話し方は心地よい。': 'Q8',
        'Q9: この会話AIと話すのは楽しい。': 'Q9',
        'Q10: またこの会話AIと話したいと思う。': 'Q10'
    }

    df = df.rename(columns=col_mapping)
    df = df[['user_id', 'session', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']]

    # Remove rows with NaN user_id and convert to int
    df = df.dropna(subset=['user_id'])
    df['user_id'] = df['user_id'].astype(int)

    return df


def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate the Concordance Correlation Coefficient (CCC)."""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc


def calculate_metrics(user_ratings, expert_ratings, questions=None):
    """Calculate MAE, Pearson r, and CCC between user and expert ratings."""
    if questions is None:
        questions = [f'Q{i}' for i in range(1, 11)]

    results = {}

    # Flatten all ratings for overall metrics
    user_all = []
    expert_all = []

    for q in questions:
        if q in user_ratings.columns and q in expert_ratings.columns:
            user_vals = user_ratings[q].values
            expert_vals = expert_ratings[q].values

            # Remove NaN pairs
            mask = ~(np.isnan(user_vals) | np.isnan(expert_vals))
            user_vals = user_vals[mask]
            expert_vals = expert_vals[mask]

            if len(user_vals) > 1:
                mae = np.mean(np.abs(user_vals - expert_vals))
                r, p_value = stats.pearsonr(user_vals, expert_vals)
                ccc = concordance_correlation_coefficient(user_vals, expert_vals)

                results[q] = {
                    'MAE': mae,
                    'Pearson_r': r,
                    'Pearson_p': p_value,
                    'CCC': ccc,
                    'N': len(user_vals)
                }

                user_all.extend(user_vals)
                expert_all.extend(expert_vals)

    # Overall metrics
    user_all = np.array(user_all)
    expert_all = np.array(expert_all)

    if len(user_all) > 1:
        results['Overall'] = {
            'MAE': np.mean(np.abs(user_all - expert_all)),
            'Pearson_r': stats.pearsonr(user_all, expert_all)[0],
            'Pearson_p': stats.pearsonr(user_all, expert_all)[1],
            'CCC': concordance_correlation_coefficient(user_all, expert_all),
            'N': len(user_all)
        }

    return results


def main():
    # File paths
    annotation_file = str(Path(__file__).resolve().parents[1] / 'data' / 'excel_files' / 'annotation_final.xlsx')
    user_assessment_file = str(Path(__file__).resolve().parents[1] / 'data' / 'excel_files' / 'user_assessment_final.xlsx')

    # Parse files
    print("Parsing annotation file...")
    expert_df = parse_annotation_file(annotation_file)
    print(f"Expert annotations: {len(expert_df)} entries")
    print(f"Expert user IDs: {sorted(expert_df['user_id'].unique())}")

    print("\nParsing user assessment file...")
    user_df = parse_user_assessment_file(user_assessment_file)
    print(f"User assessments: {len(user_df)} entries")
    print(f"User IDs: {sorted(user_df['user_id'].unique())}")

    # Filter out corrupted data
    print("\nFiltering out corrupted videos/annotations...")
    expert_df = expert_df[~expert_df['user_id'].isin(BAD_ID)]
    user_df = user_df[~user_df['user_id'].isin(BAD_ID)]
    expert_df = expert_df[~expert_df.apply(lambda r: (r['user_id'], r['session']) in BAD_VIDEOS, axis=1)]
    user_df = user_df[~user_df.apply(lambda r: (r['user_id'], r['session']) in BAD_VIDEOS, axis=1)]
    print(f"After filtering - Expert annotations: {len(expert_df)}, User assessments: {len(user_df)}")

    # Find common user IDs
    common_users = set(expert_df['user_id'].unique()) & set(user_df['user_id'].unique())
    print(f"\nCommon user IDs: {sorted(common_users)}")
    print(f"Number of common users: {len(common_users)}")

    if len(common_users) == 0:
        print("\nNo common users found between annotation and user assessment files!")
        print("Please check the user ID mapping.")
        return

    # Filter to common users
    expert_df = expert_df[expert_df['user_id'].isin(common_users)]
    user_df = user_df[user_df['user_id'].isin(common_users)]

    # Merge on user_id and session
    merged = pd.merge(
        user_df,
        expert_df,
        on=['user_id', 'session'],
        suffixes=('_user', '_expert')
    )
    print(f"\nMatched entries: {len(merged)}")

    if len(merged) == 0:
        print("No matching (user_id, session) pairs found!")
        return

    # Prepare data for metrics calculation
    user_ratings = merged[[f'Q{i}_user' for i in range(1, 11)]].rename(
        columns={f'Q{i}_user': f'Q{i}' for i in range(1, 11)}
    )
    expert_ratings = merged[[f'Q{i}_expert' for i in range(1, 11)]].rename(
        columns={f'Q{i}_expert': f'Q{i}' for i in range(1, 11)}
    )

    # Calculate metrics
    print("\n" + "="*80)
    print("COMPARISON RESULTS: Expert Annotations vs User Self-Reports")
    print("="*80)

    results = calculate_metrics(user_ratings, expert_ratings)

    # Print results in a nice table
    print(f"\n{'Question':<12} {'MAE':>8} {'Pearson r':>12} {'p-value':>12} {'CCC':>10} {'N':>6}")
    print("-"*62)

    questions = [f'Q{i}' for i in range(1, 11)] + ['Overall']
    for q in questions:
        if q in results:
            r = results[q]
            p_str = f"{r['Pearson_p']:.4f}" if r['Pearson_p'] >= 0.0001 else "<0.0001"
            print(f"{q:<12} {r['MAE']:>8.3f} {r['Pearson_r']:>12.3f} {p_str:>12} {r['CCC']:>10.3f} {r['N']:>6}")

    # Group analysis by question categories
    print("\n" + "="*80)
    print("ANALYSIS BY QUESTION CATEGORY")
    print("="*80)

    categories = {
        'Familiarity (Q1-Q2)': ['Q1', 'Q2'],
        'Social Penetration (Q3-Q4)': ['Q3', 'Q4'],
        'Long-term Memory (Q5-Q6)': ['Q5', 'Q6'],
        'Conversational System (Q7-Q8)': ['Q7', 'Q8'],
        'Enjoyment/Reuse (Q9-Q10)': ['Q9', 'Q10']
    }

    category_results = {}
    for cat_name, cat_qs in categories.items():
        # Average the two questions in each category for each session
        cat_user_avg = user_ratings[cat_qs].mean(axis=1).values
        cat_expert_avg = expert_ratings[cat_qs].mean(axis=1).values

        # Remove NaN pairs
        mask = ~(np.isnan(cat_user_avg) | np.isnan(cat_expert_avg))
        cat_user_avg = cat_user_avg[mask]
        cat_expert_avg = cat_expert_avg[mask]

        if len(cat_user_avg) > 1:
            mae = np.mean(np.abs(cat_user_avg - cat_expert_avg))
            r, p = stats.pearsonr(cat_user_avg, cat_expert_avg)
            ccc = concordance_correlation_coefficient(cat_user_avg, cat_expert_avg)
            print(f"\n{cat_name}:")
            print(f"  MAE: {mae:.3f}, Pearson r: {r:.3f} (p={p:.4f}), CCC: {ccc:.3f}, N={len(cat_user_avg)}")
            category_results[cat_name] = {
                'MAE': mae,
                'Pearson_r': r,
                'Pearson_p': p,
                'CCC': ccc,
                'N': len(cat_user_avg)
            }

    # Analysis by session
    print("\n" + "="*80)
    print("ANALYSIS BY SESSION")
    print("="*80)

    session_results = {}
    for session_num in sorted(merged['session'].unique()):
        session_mask = merged['session'] == session_num
        session_user = user_ratings[session_mask]
        session_expert = expert_ratings[session_mask]

        # Flatten all Q1-Q10 ratings for this session
        user_vals = session_user.values.flatten()
        expert_vals = session_expert.values.flatten()

        # Remove NaN pairs
        mask = ~(np.isnan(user_vals) | np.isnan(expert_vals))
        user_vals = user_vals[mask]
        expert_vals = expert_vals[mask]

        if len(user_vals) > 1:
            mae = np.mean(np.abs(user_vals - expert_vals))
            r, p = stats.pearsonr(user_vals, expert_vals)
            ccc = concordance_correlation_coefficient(user_vals, expert_vals)
            session_results[session_num] = {
                'Session': session_num,
                'MAE': mae,
                'Pearson_r': r,
                'Pearson_p': p,
                'CCC': ccc,
                'N': len(user_vals)
            }
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            print(f"Session {session_num:>2}: MAE={mae:.3f}, Pearson r={r:.3f} (p={p_str}), CCC={ccc:.3f}, N={len(user_vals)}")

    # Save session results to CSV
    session_output_file = str(Path(__file__).resolve().parent / 'session_comparison_results.csv')
    session_df = pd.DataFrame(session_results).T
    session_df = session_df.reset_index(drop=True)
    session_df['Session'] = session_df['Session'].astype(int)
    session_df = session_df[['Session', 'MAE', 'Pearson_r', 'Pearson_p', 'CCC', 'N']]
    session_df.to_csv(session_output_file, index=False)
    print(f"\nSession results saved to: {session_output_file}")

    # Analysis by session AND category
    print("\n" + "="*80)
    print("ANALYSIS BY SESSION AND CATEGORY")
    print("="*80)

    session_category_results = []
    for session_num in sorted(merged['session'].unique()):
        if session_num > 10:  # Skip unexpected sessions
            continue
        session_mask = merged['session'] == session_num
        session_user = user_ratings[session_mask]
        session_expert = expert_ratings[session_mask]

        print(f"\nSession {session_num}:")
        for cat_name, cat_qs in categories.items():
            # Average the two questions in each category
            cat_user_avg = session_user[cat_qs].mean(axis=1).values
            cat_expert_avg = session_expert[cat_qs].mean(axis=1).values

            # Remove NaN pairs
            mask = ~(np.isnan(cat_user_avg) | np.isnan(cat_expert_avg))
            cat_user_avg = cat_user_avg[mask]
            cat_expert_avg = cat_expert_avg[mask]

            if len(cat_user_avg) > 1:
                mae = np.mean(np.abs(cat_user_avg - cat_expert_avg))
                r, p = stats.pearsonr(cat_user_avg, cat_expert_avg)
                ccc = concordance_correlation_coefficient(cat_user_avg, cat_expert_avg)
                session_category_results.append({
                    'Session': session_num,
                    'Category': cat_name,
                    'MAE': mae,
                    'Pearson_r': r,
                    'Pearson_p': p,
                    'CCC': ccc,
                    'N': len(cat_user_avg)
                })
                p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
                print(f"  {cat_name}: MAE={mae:.3f}, r={r:.3f} (p={p_str}), CCC={ccc:.3f}")

    # Save session-category results to CSV
    session_category_output_file = str(Path(__file__).resolve().parent / 'session_category_comparison_results.csv')
    session_category_df = pd.DataFrame(session_category_results)
    session_category_df.to_csv(session_category_output_file, index=False)
    print(f"\nSession-category results saved to: {session_category_output_file}")

    # Save detailed results
    output_file = str(Path(__file__).resolve().parent / 'annotation_comparison_results.csv')
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_file)
    print(f"\nDetailed results saved to: {output_file}")

    # Save category results
    category_output_file = str(Path(__file__).resolve().parent / 'category_comparison_results.csv')
    category_df = pd.DataFrame(category_results).T
    category_df.to_csv(category_output_file)
    print(f"Category results saved to: {category_output_file}")

    # Also save the merged data for further analysis
    merged_file = str(Path(__file__).resolve().parent / 'merged_ratings.csv')
    merged.to_csv(merged_file, index=False)
    print(f"Merged ratings saved to: {merged_file}")


if __name__ == "__main__":
    main()

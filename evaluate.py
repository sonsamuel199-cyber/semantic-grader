"""
Evaluation Script — Chapter 3.3.3 (Evaluation Method)
--------------------------------------------------------
Computes the statistical measures used to compare the system's generated
scores against human-assigned grades:

    - Mean Absolute Error (MAE)
    - Pearson Correlation Coefficient (r)
    - Paired Samples T-Test (t, p-value)

This is a quick local sanity check while you're developing. For the actual
thesis defense, re-run the full statistical analysis in Jamovi as specified
in Chapter 3.3.3.

Usage:
    python evaluate.py results.csv

Expected CSV columns (both on the same 0-100 scale):
    system_score, human_grade
"""

import sys
import pandas as pd
from scipy import stats


def evaluate(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    required = {"system_score", "human_grade"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    mae = (df["system_score"] - df["human_grade"]).abs().mean()
    pearson_r, pearson_p = stats.pearsonr(df["system_score"], df["human_grade"])
    t_stat, t_p = stats.ttest_rel(df["system_score"], df["human_grade"])

    print(f"n = {len(df)}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Pearson Correlation (r):   {pearson_r:.3f}  (p = {pearson_p:.4f})")
    print(f"Paired Samples T-Test:     t = {t_stat:.3f}  (p = {t_p:.4f})")
    print()
    print("Note: this is a quick sanity check only. Re-run the full analysis")
    print("in Jamovi for your thesis defense, per Chapter 3.3.3.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <results.csv>")
        sys.exit(1)
    evaluate(sys.argv[1])

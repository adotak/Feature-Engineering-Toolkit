"""
main.py — Feature Engineering Toolkit
======================================
End-to-end pipeline: load data -> engineer features -> select features -> visualize.

Usage:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_dataset, describe_dataset
from src.feature_engineering import (
    apply_log_transform,
    apply_label_encoding,
    apply_one_hot_encoding,
    create_interaction_feature,
    create_polynomial_features,
)
from src.feature_selection import (
    chi_square_selection,
    rfe_selection,
    lasso_selection,
)
from src.visualization import (
    plot_log_transform,
    plot_chi_square_scores,
    plot_rfe_ranking,
    plot_lasso_coefficients,
    plot_correlation_heatmap,
)


DIVIDER = "=" * 64


def header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)



def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────
    header("LOADING DATASET")
    df = load_dataset()
    summary = describe_dataset(df)
    print(f"Shape : {summary['shape']}")
    print(f"Nulls : {sum(summary['null_counts'].values())} total missing values")
    print(df.head().to_string(index=False))

    # ======================================================================
    #  PART 1 -- FEATURE ENGINEERING
    # ======================================================================

    # 1. Log transformation
    header("1 - LOG TRANSFORMATION  (Family_Income)")
    df = apply_log_transform(df, "Family_Income")
    print(df[["Student_ID", "Family_Income", "Log_Family_Income"]].head(10).to_string(index=False))
    print("  -> Compresses wide-range values to reduce outlier influence.")

    # 2. Label encoding
    header("2 - LABEL ENCODING  (Gender, Final_Result)")
    df, mappings = apply_label_encoding(df, ["Gender", "Final_Result"])
    for col, mapping in mappings.items():
        print(f"  {col}: {mapping}")
    print(df[["Student_ID", "Gender", "Gender_Encoded",
              "Final_Result", "Final_Result_Encoded"]].head(10).to_string(index=False))

    # 3. One-hot encoding
    header("3 - ONE-HOT ENCODING  (Department)")
    df = apply_one_hot_encoding(df, "Department")
    dept_cols = [c for c in df.columns if c.startswith("Department_")]
    print(df[["Student_ID", "Department"] + dept_cols].head(10).to_string(index=False))
    print("  -> Creates binary columns for each category -- no false ordering.")

    # 4. Feature interaction
    header("4 - FEATURE INTERACTION  (Study_Hours x Attendance)")
    df = create_interaction_feature(df, "Study_Hours", "Attendance")
    print(df[["Student_ID", "Study_Hours", "Attendance",
              "Study_Hours_x_Attendance"]].head(10).to_string(index=False))
    print("  -> Captures the joint effect of effort and regularity.")

    # 5. Polynomial features
    header("5 - POLYNOMIAL FEATURES  (Study_Hours, Previous_GPA - degree 2)")
    df = create_polynomial_features(df, ["Study_Hours", "Previous_GPA"], degree=2)
    poly_cols = [c for c in df.columns if "pow" in c or "_x_Previous" in c]
    print(df[["Student_ID", "Study_Hours", "Previous_GPA"] + poly_cols].head(10).to_string(index=False))
    print("  -> Enables linear models to capture nonlinear relationships.")

    # ======================================================================
    #  PART 2 -- FEATURE SELECTION
    # ======================================================================

    # Prepare feature matrix
    feature_cols = [
        "Study_Hours", "Family_Income", "Gender_Encoded",
        "Department_CS", "Department_IT", "Department_SE",
        "Attendance", "Previous_GPA",
    ]
    X = df[feature_cols]
    y = df["Final_Result_Encoded"]

    # 6. Chi-Square (Filter)
    header("6 - CHI-SQUARE TEST  (Filter Method)")
    chi_df, chi_selected = chi_square_selection(X, y, k=3)
    print(chi_df.to_string(index=False))
    print(f"\n  Top 3 selected: {chi_selected}")

    # 7. RFE (Wrapper)
    header("7 - RECURSIVE FEATURE ELIMINATION  (Wrapper Method)")
    rfe_df, rfe_selected = rfe_selection(X, y, n_features=3)
    print(rfe_df.to_string(index=False))
    print(f"\n  Top 3 selected: {rfe_selected}")

    # 8. LASSO (Embedded)
    header("8 - LASSO REGRESSION  (Embedded Method)")
    lasso_df, retained, removed = lasso_selection(X, y, alpha=0.01)
    print(lasso_df.to_string(index=False))
    print(f"\n  Retained: {retained}")
    print(f"  Removed : {removed if removed else 'None at alpha=0.01'}")

    # ======================================================================
    #  PART 3 -- VISUALIZATIONS
    # ======================================================================
    header("GENERATING VISUALIZATIONS")

    charts = [
        ("Log transform comparison", plot_log_transform(df)),
        ("Chi-Square scores", plot_chi_square_scores(chi_df)),
        ("RFE ranking", plot_rfe_ranking(rfe_df)),
        ("LASSO coefficients", plot_lasso_coefficients(lasso_df)),
        ("Correlation heatmap", plot_correlation_heatmap(df)),
    ]

    for name, path in charts:
        print(f"  [OK] {name:40s} {path}")

    # ── Done ─────────────────────────────────────────────────────────────
    header("PIPELINE COMPLETE")
    print("  All charts saved to outputs/\n")


if __name__ == "__main__":
    main()

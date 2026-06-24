"""
feature_selection.py — Filter, Wrapper, and Embedded feature-selection methods.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression, Lasso


# ── Filter Method — Chi-Square ───────────────────────────────────────────────

def chi_square_selection(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Rank features by Chi-Square statistic and select the top *k*.

    Chi-Square measures the statistical dependence between each feature
    and the target.  It requires non-negative inputs, so any column with
    negative values is shifted automatically.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A scores DataFrame (Feature, Chi2_Score, P_Value, Rank) and the
        list of top-*k* selected feature names.
    """
    X_abs = X.copy()
    for col in X_abs.columns:
        col_min = X_abs[col].min()
        if col_min < 0:
            X_abs[col] = X_abs[col] - col_min

    scores, pvalues = chi2(X_abs, y)

    scores_df = (
        pd.DataFrame({
            "Feature": X.columns,
            "Chi2_Score": scores,
            "P_Value": pvalues,
        })
        .sort_values("Chi2_Score", ascending=False)
        .reset_index(drop=True)
    )
    scores_df["Rank"] = scores_df.index + 1

    selector = SelectKBest(chi2, k=k)
    selector.fit(X_abs, y)
    selected = list(X.columns[selector.get_support()])

    return scores_df, selected


# ── Wrapper Method — Recursive Feature Elimination ───────────────────────────

def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 3,
    estimator=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply RFE with a Logistic Regression estimator (default).

    RFE iteratively removes the least important feature according to the
    estimator's coefficients until *n_features* remain.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A ranking DataFrame and the list of selected feature names.
    """
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, random_state=42)

    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)

    ranking_df = (
        pd.DataFrame({
            "Feature": X.columns,
            "RFE_Rank": rfe.ranking_,
            "Selected": rfe.support_,
        })
        .sort_values("RFE_Rank")
    )

    selected = list(X.columns[rfe.support_])
    return ranking_df, selected


# ── Embedded Method — LASSO ──────────────────────────────────────────────────

def lasso_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.01,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Fit a LASSO model and identify retained / removed features.

    LASSO's L1 penalty drives insignificant coefficients to exactly zero,
    performing feature selection as part of the training process.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        Coefficient DataFrame, list of retained features, list of removed
        features.
    """
    model = Lasso(alpha=alpha, random_state=42, max_iter=5000)
    model.fit(X, y)

    coef_df = (
        pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_,
        })
        .sort_values("Coefficient", key=abs, ascending=False)
    )

    retained = coef_df.loc[coef_df["Coefficient"] != 0, "Feature"].tolist()
    removed = coef_df.loc[coef_df["Coefficient"] == 0, "Feature"].tolist()

    return coef_df, retained, removed

"""
feature_engineering.py — Pure-function transformations for feature engineering.

Every public function follows the pattern:
    take a DataFrame → return a *new* DataFrame (no in-place mutation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures


# ── Numeric transforms ───────────────────────────────────────────────────────

def apply_log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply ``log(x + 1)`` to *column* and store result as ``Log_<column>``.

    Adding 1 prevents ``log(0)`` issues for zero-valued entries.
    """
    df = df.copy()
    new_col = f"Log_{column}"
    df[new_col] = np.log(df[column] + 1)
    return df


# ── Categorical encoding ────────────────────────────────────────────────────

def apply_label_encoding(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Label-encode one or more columns.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Modified DataFrame and a mapping ``{column: {label: int, ...}}``.
    """
    df = df.copy()
    mappings: dict[str, dict] = {}

    for col in columns:
        le = LabelEncoder()
        encoded_col = f"{col}_Encoded"
        df[encoded_col] = le.fit_transform(df[col])
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, mappings


def apply_one_hot_encoding(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One-hot encode *column* using ``pd.get_dummies``."""
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column).astype(int)
    df = pd.concat([df, dummies], axis=1)
    return df


# ── Interaction & polynomial features ───────────────────────────────────────

def create_interaction_feature(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Multiply two columns to capture their joint effect.

    Parameters
    ----------
    name : str, optional
        Name for the new column.  Defaults to ``<col_a>_x_<col_b>``.
    """
    df = df.copy()
    feature_name = name or f"{col_a}_x_{col_b}"
    df[feature_name] = df[col_a] * df[col_b]
    return df


def create_polynomial_features(
    df: pd.DataFrame,
    columns: list[str],
    degree: int = 2,
) -> pd.DataFrame:
    """Generate polynomial and interaction terms up to *degree*.

    New columns are appended with descriptive suffixes (e.g. ``^2``, ``_x_``).
    """
    df = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    transformed = poly.fit_transform(df[columns].values)
    names = poly.get_feature_names_out(columns)

    # Make names more readable
    readable = []
    for n in names:
        n = n.replace(" ", "_x_")
        for col in columns:
            n = n.replace(f"{col}^", f"{col}_pow")
        readable.append(n)

    poly_df = pd.DataFrame(transformed, columns=readable, index=df.index)

    # Only add the *new* columns (skip the original ones)
    new_cols = [c for c in readable if c not in columns]
    for col in new_cols:
        df[col] = poly_df[col]

    return df

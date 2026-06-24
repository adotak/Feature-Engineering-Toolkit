"""
visualization.py — Generate publication-quality charts for every pipeline stage.

All plots are saved to the ``outputs/`` directory as PNG files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style defaults ───────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {
    "primary": "#4361EE",
    "secondary": "#7209B7",
    "accent": "#F72585",
    "positive": "#06D6A0",
    "negative": "#EF476F",
    "neutral": "#118AB2",
}
OUTPUT_DIR = Path("outputs")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Log-transform comparison ─────────────────────────────────────────────

def plot_log_transform(
    df: pd.DataFrame,
    original_col: str = "Family_Income",
    log_col: str = "Log_Family_Income",
) -> Path:
    """Side-by-side histograms: original vs. log-transformed distribution."""
    _ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original
    axes[0].hist(
        df[original_col], bins=20, color=PALETTE["primary"],
        edgecolor="white", alpha=0.85,
    )
    axes[0].set_title("Original Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(original_col)
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df[original_col].mean(), color=PALETTE["accent"],
                    linestyle="--", linewidth=1.5, label="Mean")
    axes[0].legend()

    # Log-transformed
    axes[1].hist(
        df[log_col], bins=20, color=PALETTE["secondary"],
        edgecolor="white", alpha=0.85,
    )
    axes[1].set_title("After Log Transform", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(log_col)
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(df[log_col].mean(), color=PALETTE["accent"],
                    linestyle="--", linewidth=1.5, label="Mean")
    axes[1].legend()

    fig.suptitle(
        "Effect of Log Transformation on Income Distribution",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out = OUTPUT_DIR / "log_transform_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 2. Chi-Square feature scores ────────────────────────────────────────────

def plot_chi_square_scores(scores_df: pd.DataFrame) -> Path:
    """Horizontal bar chart of Chi-Square scores ranked by importance."""
    _ensure_output_dir()

    ordered = scores_df.sort_values("Chi2_Score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["accent"] if i >= len(ordered) - 3 else PALETTE["primary"]
              for i in range(len(ordered))]
    ax.barh(ordered["Feature"], ordered["Chi2_Score"], color=colors,
            edgecolor="white", height=0.6)
    ax.set_xlabel("Chi-Square Score")
    ax.set_title("Feature Importance — Chi-Square (Filter Method)",
                 fontsize=14, fontweight="bold")

    # Annotate values
    for idx, (score, feature) in enumerate(
        zip(ordered["Chi2_Score"], ordered["Feature"])
    ):
        ax.text(score + max(ordered["Chi2_Score"]) * 0.01, idx,
                f"{score:,.1f}", va="center", fontsize=10)

    fig.tight_layout()
    out = OUTPUT_DIR / "chi_square_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 3. RFE ranking ──────────────────────────────────────────────────────────

def plot_rfe_ranking(ranking_df: pd.DataFrame) -> Path:
    """Bar chart of RFE feature rankings (lower = better)."""
    _ensure_output_dir()

    ordered = ranking_df.sort_values("RFE_Rank", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["positive"] if sel else PALETTE["neutral"]
              for sel in ordered["Selected"]]
    ax.barh(ordered["Feature"], ordered["RFE_Rank"], color=colors,
            edgecolor="white", height=0.6)
    ax.set_xlabel("RFE Rank (1 = most important)")
    ax.set_title("Feature Ranking — Recursive Feature Elimination (Wrapper Method)",
                 fontsize=14, fontweight="bold")
    ax.invert_xaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["positive"], label="Selected"),
        Patch(facecolor=PALETTE["neutral"], label="Eliminated"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    out = OUTPUT_DIR / "rfe_ranking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 4. LASSO coefficients ───────────────────────────────────────────────────

def plot_lasso_coefficients(coef_df: pd.DataFrame) -> Path:
    """Horizontal bar chart of LASSO coefficients with a zero-line."""
    _ensure_output_dir()

    ordered = coef_df.sort_values("Coefficient")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["positive"] if c > 0 else
              (PALETTE["negative"] if c < 0 else "#AAAAAA")
              for c in ordered["Coefficient"]]
    ax.barh(ordered["Feature"], ordered["Coefficient"], color=colors,
            edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Feature Coefficients — LASSO Regression (Embedded Method)",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    out = OUTPUT_DIR / "lasso_coefficients.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 5. Correlation heatmap ──────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Correlation heatmap of all numeric columns."""
    _ensure_output_dir()

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap",
                 fontsize=15, fontweight="bold", pad=15)

    fig.tight_layout()
    out = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

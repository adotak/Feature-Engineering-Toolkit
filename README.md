# Feature Engineering Toolkit

A modular Python toolkit that demonstrates **feature engineering** and **feature selection** techniques on a student performance dataset. Built as a reusable pipeline, it transforms raw data into model-ready features and evaluates each feature's predictive importance using statistical and machine-learning-based methods.

## What It Does

```
Raw CSV  →  Feature Engineering  →  Feature Selection  →  Visualizations
```

**Feature Engineering** (transforming data):
| Technique | Purpose |
|---|---|
| Log Transformation | Compress skewed distributions (e.g., income) |
| Label Encoding | Convert binary categories to 0/1 |
| One-Hot Encoding | Represent nominal categories without false ordering |
| Feature Interaction | Capture joint effects (Study_Hours × Attendance) |
| Polynomial Features | Enable linear models to learn nonlinear patterns |

**Feature Selection** (choosing the best features):
| Method | Type | How It Works |
|---|---|---|
| Chi-Square Test | Filter | Ranks features by statistical dependence with the target |
| Recursive Feature Elimination (RFE) | Wrapper | Iteratively removes least useful features using a model |
| LASSO Regression | Embedded | Shrinks unimportant coefficients to zero during training |

## Project Structure

```
feature-engineering-toolkit/
├── main.py                      # Entry point — runs the full pipeline
├── requirements.txt             # Dependencies
├── data/
│   └── students.csv             # Student performance dataset (100 records)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load and validate CSV data
│   ├── feature_engineering.py   # Transformation functions
│   ├── feature_selection.py     # Selection algorithms
│   └── visualization.py         # Chart generation
└── outputs/                     # Generated charts (created on run)
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/feature-engineering-toolkit.git
cd feature-engineering-toolkit

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

All charts are saved to `outputs/` automatically.

## Sample Outputs

After running the pipeline, you'll find these visualizations in `outputs/`:

- **`log_transform_comparison.png`** — Before/after income distribution
- **`chi_square_scores.png`** — Feature importance ranking (filter method)
- **`rfe_ranking.png`** — Recursive elimination results (wrapper method)
- **`lasso_coefficients.png`** — LASSO coefficient magnitudes (embedded method)
- **`correlation_heatmap.png`** — Full feature correlation matrix

## Dataset

100 student records with 7 features:

| Feature | Type | Description |
|---|---|---|
| Study_Hours | Numeric | Weekly study hours (1–10) |
| Family_Income | Numeric | Annual household income |
| Gender | Categorical | Male / Female |
| Department | Categorical | CS / IT / SE |
| Attendance | Numeric | Attendance percentage (46–100) |
| Previous_GPA | Numeric | Prior GPA (1.91–4.00) |
| Final_Result | Target | Pass / Fail |

## Tech Stack

- **Python 3.10+**
- **pandas** / **NumPy** — data manipulation
- **scikit-learn** — encoding, selection, and modeling
- **matplotlib** / **seaborn** — visualization

## License

MIT

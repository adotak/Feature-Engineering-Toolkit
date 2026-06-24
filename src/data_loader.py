"""
data_loader.py — Load and validate the student performance dataset.
"""

from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "Student_ID", "Study_Hours", "Family_Income", "Gender",
    "Department", "Attendance", "Previous_GPA", "Final_Result",
]


def load_dataset(path: str | Path = "data/students.csv") -> pd.DataFrame:
    """Read the CSV dataset and run basic validation.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file (default: ``data/students.csv``).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame ready for feature engineering.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dictionary for quick inspection.

    Returns
    -------
    dict
        Keys: ``shape``, ``dtypes``, ``null_counts``, ``numeric_summary``.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict(),
    }

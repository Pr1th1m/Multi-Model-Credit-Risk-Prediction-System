"""
Data loading and initial exploration utilities.
"""

import pandas as pd
from src.config import DATA_PATH, TARGET_COLUMN, COLUMN_MAP


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the German Credit dataset and print a quick summary."""
    df = pd.read_csv(path)
    print("=" * 60)
    print("DATA LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Shape           : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print(f"  Duplicates      : {df.duplicated().sum()}")
    print()
    return df


def print_target_distribution(df: pd.DataFrame) -> None:
    """Print class distribution of the target column."""
    counts = df[TARGET_COLUMN].value_counts()
    pcts = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    print("TARGET DISTRIBUTION")
    print("-" * 40)
    for label in counts.index:
        status = "Good Credit" if label == 1 else "Bad Credit"
        print(f"  {label} ({status}): {counts[label]:>4}  ({pcts[label]:.1f}%)")
    print()


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics with English column names."""
    renamed = df.rename(columns=COLUMN_MAP)
    print("FEATURE SUMMARY")
    print("-" * 40)
    print(renamed.describe().T.to_string())
    print()

"""
Data preprocessing: feature/target split, scaling, and train/test split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def prepare_data(df: pd.DataFrame):
    """
    Split features and target, scale features, and create train/test sets.

    Returns:
        X_train, X_test, y_train, y_test  (all NumPy arrays)
        feature_names (list of str)
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)

    # Train / Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("PREPROCESSING COMPLETE")
    print("-" * 40)
    print(f"  Training set : {X_train.shape[0]} samples")
    print(f"  Test set     : {X_test.shape[0]} samples")
    print(f"  Features     : {X_train.shape[1]}")
    print()

    return X_train, X_test, y_train, y_test, feature_names

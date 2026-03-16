"""
Configuration for the Credit Risk Prediction Pipeline.
Constants, column mappings, hyperparameter grids, and paths.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "german_credit_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
ROC_DIR = os.path.join(RESULTS_DIR, "roc_curves")

# ──────────────────────────────────────────────
# Data settings
# ──────────────────────────────────────────────
TARGET_COLUMN = "kredit"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

# German → English column name mapping (for readability)
COLUMN_MAP = {
    "laufkont": "checking_account",
    "laufzeit": "duration_months",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "credit_amount",
    "sparkont": "savings_account",
    "beszeit": "employment_years",
    "rate": "installment_rate",
    "famges": "personal_status",
    "buerge": "guarantors",
    "wohnzeit": "residence_years",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installments",
    "wohn": "housing",
    "bishkred": "existing_credits",
    "beruf": "job",
    "pers": "dependents",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

# ──────────────────────────────────────────────
# Hyperparameter grids (kept small for speed)
# ──────────────────────────────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
}

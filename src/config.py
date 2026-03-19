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
SHAP_DIR = os.path.join(RESULTS_DIR, "shap_plots")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "Credit_Risk_Multi_Model"

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
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000, 2000],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear", "sigmoid"],
        "gamma": ["scale", "auto", 0.1, 1],
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7],
    },
}

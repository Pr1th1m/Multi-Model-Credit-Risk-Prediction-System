"""
Pipeline orchestrator — trains, tunes, evaluates, and compares all models.
"""

import time
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.config import RANDOM_STATE, CV_FOLDS, RESULTS_DIR
from src.data_loader import load_data, print_target_distribution, print_feature_summary
from src.preprocessing import prepare_data
from src.models import get_models
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    save_comparison_table,
)

import os


def run_pipeline():
    """End-to-end pipeline: load → preprocess → train → evaluate → compare."""
    print("\n" + "=" * 60)
    print("  MULTI-MODEL CREDIT RISK PREDICTION PIPELINE")
    print("=" * 60 + "\n")

    # ── 1. Load data ─────────────────────────────
    df = load_data()
    print_target_distribution(df)
    print_feature_summary(df)

    # ── 2. Preprocess ────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    # ── 3. Train & evaluate each model ───────────
    models = get_models()
    results = []

    for name, (estimator, param_grid) in models.items():
        print(f"\n{'=' * 60}")
        print(f"  TRAINING: {name}")
        print(f"{'=' * 60}")

        start = time.time()

        # GridSearchCV for hyperparameter tuning
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=CV_FOLDS,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        elapsed = time.time() - start

        best_model = grid.best_estimator_
        print(f"  Best params : {grid.best_params_}")
        print(f"  Best CV AUC : {grid.best_score_:.4f}")
        print(f"  Train time  : {elapsed:.1f}s")

        # Evaluate on test set
        metrics = evaluate_model(name, best_model, X_test, y_test)
        results.append(metrics)

        # Predictions & probabilities for plots
        y_pred = best_model.predict(X_test)
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_proba = best_model.decision_function(X_test)
        else:
            y_proba = y_pred.astype(float)

        # Save plots
        plot_confusion_matrix(name, y_test, y_pred)
        plot_roc_curve(name, y_test, y_proba)

    # ── 4. Compare all models ────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_results = save_comparison_table(results, comparison_path)
    print()
    print(df_results.to_string(index=False))

    # ── 5. Best model ────────────────────────────
    best = df_results.iloc[0]
    print(f"\n{'=' * 60}")
    print(f"  🏆 BEST MODEL: {best['Model']}")
    print(f"     AUC-ROC = {best['AUC-ROC']}")
    print(f"     F1      = {best['F1-Score']}")
    print(f"     Accuracy= {best['Accuracy']}")
    print(f"{'=' * 60}\n")

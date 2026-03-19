"""
Pipeline orchestrator — trains, tunes, evaluates, and compares all models.
"""

import time
import numpy as np
from sklearn.model_selection import cross_val_score
import optuna
import pandas as pd

from src.config import RANDOM_STATE, CV_FOLDS, RESULTS_DIR, MLFLOW_EXPERIMENT_NAME, COLUMN_MAP
from src.data_loader import load_data, print_target_distribution, print_feature_summary
from src.preprocessing import prepare_data
from src.models import get_models
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    save_comparison_table,
    plot_shap_summary,
)

import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost


def run_pipeline():
    print("\n" + "=" * 60)
    print("  MULTI-MODEL CREDIT RISK PREDICTION PIPELINE")
    print("=" * 60 + "\n")

    # ── 1. Load data ─────────────────────────────
    df = load_data()
    print_target_distribution(df)
    print_feature_summary(df)

    # ── 2. Preprocess ────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    # Setup MLflow Experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── 3. Train & evaluate each model ───────────
    models = get_models()
    results = []

    for name, (estimator, param_grid) in models.items():
        print(f"\n{'=' * 60}")
        print(f"  TRAINING: {name}")
        print(f"{'=' * 60}")

        with mlflow.start_run(run_name=name):
            start = time.time()
            # Optuna for hyperparameter tuning
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            from sklearn.base import clone
            def objective(trial):
                params = {}
                for param_name, param_values in param_grid.items():
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                
                model = clone(estimator)
                model.set_params(**params)
                
                scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=CV_FOLDS, 
                    scoring="roc_auc", 
                    n_jobs=-1
                )
                return np.mean(scores)
            
            study = optuna.create_study(direction="maximize")

            # Calculate upper bound on combinations based on grid
            max_trials = 1
            for vals in param_grid.items():
                max_trials *= len(vals[1])
            
            # Limit trials to 20 or max combinations for small grids
            n_trials = min(20, max_trials)
            study.optimize(objective, n_trials=n_trials)
            
            elapsed = time.time() - start
    
            best_params = study.best_params
            best_model = clone(estimator).set_params(**best_params)
            best_model.fit(X_train, y_train)

            print(f"  Best params : {best_params}")
            print(f"  Best CV AUC : {study.best_value:.4f}")
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
            cm_path = plot_confusion_matrix(name, y_test, y_pred)
            roc_path = plot_roc_curve(name, y_test, y_proba)
            
            # Generate SHAP summary
            X_test_df = pd.DataFrame(X_test, columns=feature_names).rename(columns=COLUMN_MAP)
            try:
                shap_path = plot_shap_summary(name, best_model, X_test_df)
            except Exception as e:
                shap_path = None
                print(f"  ✗ SHAP summary skipped for {name}: {e}")
            
            # ── Log to MLflow ─────────────────────────
            # Log best hyperparameters
            mlflow.log_params(best_params)
            
            # Log plots to MLflow artifacts
            mlflow.log_artifact(cm_path, "plots")
            mlflow.log_artifact(roc_path, "plots")
            if shap_path:
                mlflow.log_artifact(shap_path, "plots")
            
            # Log metrics (Precision, Recall, F1, Accuracy, AUC)
            # Remove string key 'Model' to just log numerical metrics
            mlflow_metrics = {k: v for k, v in metrics.items() if k != "Model"}
            mlflow.log_metrics(mlflow_metrics)
            
            # Log the fitted model itself
            if name == "XGBoost":
                mlflow.xgboost.log_model(best_model, "model")
            elif name == "LightGBM":
                mlflow.lightgbm.log_model(best_model, "model")
            elif name == "CatBoost":
                mlflow.catboost.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")

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

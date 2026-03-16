"""
Model evaluation: metrics computation, confusion matrix, ROC curve plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.config import CONFUSION_DIR, ROC_DIR


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Evaluate a trained model and return a metrics dictionary.
    Also prints the classification report.
    """
    y_pred = model.predict(X_test)

    # Probability scores for AUC-ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'─' * 50}")
    print(f"  {name} — Classification Report")
    print(f"{'─' * 50}")
    print(classification_report(y_test, y_pred, target_names=["Bad (0)", "Good (1)"]))

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "AUC-ROC": round(auc, 4),
    }


def plot_confusion_matrix(name: str, y_test, y_pred, save_dir: str = CONFUSION_DIR):
    """Save a confusion matrix heatmap as PNG."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Bad (0)", "Good (1)"],
        yticklabels=["Bad (0)", "Good (1)"],
        ax=ax,
    )
    ax.set_title(f"{name} — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    filepath = os.path.join(save_dir, f"{name.replace(' ', '_')}_cm.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ Confusion matrix saved → {filepath}")


def plot_roc_curve(name: str, y_test, y_proba, save_dir: str = ROC_DIR):
    """Save an ROC curve plot as PNG."""
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} — ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    filepath = os.path.join(save_dir, f"{name.replace(' ', '_')}_roc.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ ROC curve saved → {filepath}")


def save_comparison_table(results: list[dict], save_path: str) -> pd.DataFrame:
    """Save all model results as a CSV and return the DataFrame."""
    df = pd.DataFrame(results)
    df = df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n✓ Comparison table saved → {save_path}")
    return df

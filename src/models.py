"""
Model definitions for the Credit Risk Prediction Pipeline.
Returns a dictionary of {name: (model_instance, param_grid)}.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import PARAM_GRIDS, RANDOM_STATE


def get_models() -> dict:
    """
    Return a dict mapping model name → (estimator, param_grid).
    Class-weight balancing is applied where supported.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            PARAM_GRIDS["Logistic Regression"],
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            PARAM_GRIDS["Decision Tree"],
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            PARAM_GRIDS["Random Forest"],
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                scale_pos_weight=2.33,   # ≈ 700/300 ratio
                verbosity=0,
            ),
            PARAM_GRIDS["XGBoost"],
        ),
        "LightGBM": (
            LGBMClassifier(
                random_state=RANDOM_STATE,
                is_unbalance=True,
                verbose=-1,
            ),
            PARAM_GRIDS["LightGBM"],
        ),
        "KNN": (
            KNeighborsClassifier(),
            PARAM_GRIDS["KNN"],
        ),
        "SVM": (
            SVC(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                probability=True,       # needed for ROC curves
            ),
            PARAM_GRIDS["SVM"],
        ),
        "CatBoost": (
            CatBoostClassifier(
                random_state=RANDOM_STATE,
                auto_class_weights="Balanced",
                verbose=0,
            ),
            PARAM_GRIDS["CatBoost"],
        ),
    }
    return models

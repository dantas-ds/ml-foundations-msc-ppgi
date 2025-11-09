import numpy as np

from typing import Any, Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def fit_predict_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 42,
    max_iter: int = 1000,
) -> Dict[str, Any]:

    """
    Train logistic regression (with standardization) and return predictions and model.
    """

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, random_state=seed, max_iter=max_iter))
    ])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    return {"model": pipe, "y_pred": y_pred}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall and F1.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

import numpy as np

from sklearn.pipeline import Pipeline
from typing import Any, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 42,
    max_iter: int = 1000,
) -> Dict[str, Any]:

    """
    Fit Logistic Regression with standardization and return model and train predictions.
    Returns
    -------
    {"model": Pipeline, "y_pred": np.ndarray}
    """

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, random_state=seed, max_iter=max_iter)),
    ])

    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    return {"model": pipe, "y_pred": y_pred}


def evaluate_classification(y_true: np.ndarray,
                            y_pred: np.ndarray) -> Dict[str, float]:

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_logreg(
    X: np.ndarray,
    y: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Train Logistic Regression on X,y and optionally evaluate on test.
    Returns
    -------
    {
      "model": Pipeline,
      "y_pred": np.ndarray,
      "metrics": Optional[Dict[str,float]],
      "train": {"y_pred": np.ndarray, "metrics": Optional[Dict[str,float]]},
      "test":  Optional[{"y_pred": np.ndarray, "metrics": Optional[Dict[str,float]]}]
    }
    """

    params = params or {}
    C = float(params.get("C", 1.0))
    seed = int(params.get("seed", 42))
    max_iter = int(params.get("max_iter", 1000))

    fit = fit_logreg(X, y, C=C, seed=seed, max_iter=max_iter)
    model, y_pred_train = fit["model"], fit["y_pred"]

    train_metrics = evaluate_classification(y, y_pred_train)

    test_block = None

    if X_test is not None:
        y_pred_test = model.predict(X_test)
        test_metrics = evaluate_classification(y_test, y_pred_test) if y_test is not None else None
        test_block = {"y_pred": y_pred_test, "metrics": test_metrics}

    return {
        "model": model,
        "y_pred": y_pred_train,
        "metrics": train_metrics,
        "train": {"y_pred": y_pred_train, "metrics": train_metrics},
        "test": test_block,
    }

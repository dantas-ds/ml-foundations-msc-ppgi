import numpy as np

from sklearn.pipeline import Pipeline
from typing import Any, Dict, Optional, Iterable
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def fit_knn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
) -> Dict[str, Any]:

    """
    Fit a K-Nearest Neighbors classifier (with standardization) and return model and train predictions.
    Returns
    -------
    {"model": Pipeline, "y_pred": np.ndarray}
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
    ])
    model.fit(X, y)
    y_pred = model.predict(X)

    return {"model": model, "y_pred": y_pred}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall and F1.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_knn(
    X: np.ndarray,
    y: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Train KNN on X,y and optionally evaluate on test.
    Returns
    -------
    {
      "model": Pipeline,
      "y_pred": np.ndarray,
      "metrics": Dict[str,float],
      "train": {"y_pred": np.ndarray, "metrics": Dict[str,float]},
      "test":  {"y_pred": np.ndarray, "metrics": Dict[str,float]} or None
    }
    """

    params = params or {}
    n_neighbors = int(params.get("n_neighbors", 5))
    weights = params.get("weights", "uniform")

    fit = fit_knn(X, y, n_neighbors=n_neighbors, weights=weights)
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


def run_knn_grid(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ks: Iterable[int] = (3, 5, 7, 9, 11),
    weights: str = "uniform",
) -> Dict[int, Dict[str, Any]]:

    """
    Evaluate multiple k values on the same split. Returns a dict[k] -> run_knn(...) result.
    """

    out: Dict[int, Dict[str, Any]] = {}

    for k in ks:
        out[k] = run_knn(
            X, y, X_test, y_test,
            params={"n_neighbors": int(k), "weights": weights}
        )
    return out

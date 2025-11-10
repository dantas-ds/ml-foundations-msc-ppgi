import numpy as np

from sklearn.svm import SVC
from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def fit_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    seed: int = 42,
) -> Dict[str, Any]:

    """
    Fit an SVM classifier with standardization and return model and train predictions.
    Returns
    -------
    {"model": Pipeline, "y_pred": np.ndarray}
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=C, gamma=gamma, random_state=seed)),
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


def run_svm(
    X: np.ndarray,
    y: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Train SVM on X,y and optionally evaluate on test.
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
    kernel = params.get("kernel", "rbf")
    C = float(params.get("C", 1.0))
    gamma = params.get("gamma", "scale")
    seed = int(params.get("seed", 42))

    fit = fit_svm(X, y, kernel=kernel, C=C, gamma=gamma, seed=seed)
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

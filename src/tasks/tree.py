import numpy as np

from typing import Any, Dict, Optional
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def fit_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int | None = None,
    criterion: str = "gini",
    seed: int = 42,
) -> Dict[str, Any]:

    """
    Fit a Decision Tree classifier and return model and train predictions.
    Returns
    -------
    {"model": DecisionTreeClassifier, "y_pred": np.ndarray}
    """

    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=seed)
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


def run_tree(
    X: np.ndarray,
    y: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Train Decision Tree on X,y and optionally evaluate on test.
    Returns
    -------
    {
      "model": DecisionTreeClassifier,
      "y_pred": np.ndarray,
      "metrics": Dict[str,float],
      "train": {"y_pred": np.ndarray, "metrics": Dict[str,float]},
      "test":  {"y_pred": np.ndarray, "metrics": Dict[str,float]} or None
    }
    """

    params = params or {}
    max_depth = params.get("max_depth", None)
    criterion = params.get("criterion", "gini")
    seed = int(params.get("seed", 42))

    fit = fit_tree(X, y, max_depth=max_depth, criterion=criterion, seed=seed)
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

import numpy as np

from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def fit_ann(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_layers: tuple[int, ...] = (10, 10),
    activation: str = "relu",
    alpha: float = 0.001,
    learning_rate_init: float = 0.01,
    max_iter: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:

    """
    Train a feedforward Artificial Neural Network (ANN) for binary 
    classification.

    Returns
    -------
    {
      "model": Pipeline,
      "y_pred": np.ndarray
    }
    """

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=seed,
        )),
    ])

    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    return {"model": pipe, "y_pred": y_pred}


def evaluate_ann(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate Accuracy, Precision, Recall, and F1-score.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_ann(
    X: np.ndarray,
    y: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Train ANN on X,y and optionally evaluate on test.

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
    hidden_layers = params.get("hidden_layers", (10, 10))
    activation = params.get("activation", "relu")
    alpha = float(params.get("alpha", 0.001))
    learning_rate_init = float(params.get("learning_rate_init", 0.01))
    max_iter = int(params.get("max_iter", 1000))
    seed = int(params.get("seed", 42))

    fit = fit_ann(
        X, y,
        hidden_layers=hidden_layers,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        seed=seed,
    )

    model, y_pred_train = fit["model"], fit["y_pred"]
    train_metrics = evaluate_ann(y, y_pred_train)

    test_block = None
    if X_test is not None:
        y_pred_test = model.predict(X_test)
        test_metrics = evaluate_ann(y_test, y_pred_test) if y_test is not None else None
        test_block = {"y_pred": y_pred_test, "metrics": test_metrics}

    return {
        "model": model,
        "y_pred": y_pred_train,
        "metrics": train_metrics,
        "train": {"y_pred": y_pred_train, "metrics": train_metrics},
        "test": test_block,
    }

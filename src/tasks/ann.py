import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from typing import Any, Dict


def fit_ann(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_layers: tuple[int, ...] = (10, 10),
    activation: str = "relu",
    alpha: float = 0.001,
    learning_rate_init: float = 0.01,
    max_iter: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:

    """
    Train a feedforward artificial neural network (ANN) for binary classification.
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
        ))
    ])

    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    return {"model": pipe, "y_pred": y_pred}


def evaluate_ann(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate accuracy, precision, recall and F1-score.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

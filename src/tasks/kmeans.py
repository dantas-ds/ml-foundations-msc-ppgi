import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from typing import Any, Dict, Optional, Tuple


def best_match_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Map predicted clusters to ground-truth labels using majority voting.
    Ensures label alignment between clustering and classes.
    """

    unique = np.unique(y_pred)
    mapped = np.empty_like(y_pred)

    for c in unique:
        mask = y_pred == c
        if not np.any(mask):
            continue

        maj = np.bincount(y_true[mask]).argmax()
        mapped[mask] = maj

    return mapped


def fit_predict_kmeans(
    X: np.ndarray,
    k: int = 2,
    seed: int = 42,
    n_init: Any = "auto",
    max_iter: int = 300,
) -> Tuple[np.ndarray, KMeans]:

    """
    Fit K-means clustering to data and return labels and trained model.
    """

    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=max_iter)
    y_pred = km.fit_predict(X)

    return y_pred, km


def evaluate_kmeans(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute confusion matrix (after majority alignment) and Adjusted Rand Index.
    """

    y_pred_mapped = best_match_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred_mapped)
    ari = adjusted_rand_score(y_true, y_pred)

    return {"confusion_matrix": cm,
            "ari": float(ari),
            "y_pred_mapped": y_pred_mapped}


def run_kmeans(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Train and optionally evaluate a K-means clustering model.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : Optional[np.ndarray], optional
        Ground truth labels for evaluation.
    params : Optional[Dict[str, Any]], optional
        KMeans hyperparameters (k, seed, n_init, max_iter).

    Returns
    -------
    Dict[str, Any]
        Results including labels, model, and metrics if y provided.
    """

    params = params or {}

    k = int(params.get("k", 2))
    seed = int(params.get("seed", 42))
    n_init = params.get("n_init", "auto")
    max_iter = int(params.get("max_iter", 300))

    y_pred, model = fit_predict_kmeans(X, k=k,
                                       seed=seed,
                                       n_init=n_init,
                                       max_iter=max_iter)

    metrics = evaluate_kmeans(y, y_pred) if y is not None else None

    return {"y_pred": y_pred, "model": model, "metrics": metrics}

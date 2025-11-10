import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree, export_text
from typing import Optional, Tuple, Iterable, Dict, Any
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def _mesh_2d(X: np.ndarray, pad: float = 1.0, n: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def _class_palette(n: int) -> np.ndarray:
    return plt.cm.Set2(np.linspace(0, 1, n))


def _is_continuous_model(model) -> bool:
    return hasattr(model, "predict_proba") or hasattr(model,
                                                      "decision_function")


def _continuous_score(model, grid: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        z = model.predict_proba(grid)[:, 1]
    else:
        z = model.decision_function(grid)
        z = (z - z.min()) / (z.max() - z.min() + 1e-12)

    return z


def _map_clusters_to_labels(y_true: np.ndarray,
                            y_pred: np.ndarray) -> np.ndarray:

    mapping: Dict[Any, Any] = {}
    for c in np.unique(y_pred):
        m = (y_pred == c)
        if np.any(m):
            mapping[c] = np.bincount(y_true[m]).argmax()

    return np.vectorize(lambda z: mapping[z])(y_pred)


def scatter_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "Dataset Scatter",
    show_ellipses: bool = True,
    alpha: float = 0.85,
    figsize: Tuple[int, int] = (7, 5),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):

    classes = np.unique(y)
    colors = _class_palette(len(classes))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, cls in enumerate(classes):
        m = (y == cls)
        ax.scatter(X[m, 0], X[m, 1], s=40, alpha=alpha, color=colors[i], label=f"Class {cls}")

        if show_ellipses and m.any():
            mx, my = X[m, 0].mean(), X[m, 1].mean()
            sx, sy = X[m, 0].std(), X[m, 1].std()
            t = np.linspace(0, 2 * np.pi, 240)
            xe, ye = mx + sx * np.cos(t), my + sy * np.sin(t)
            ax.plot(xe, ye, color=colors[i], lw=2)

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        (fig or ax.figure).savefig(save_path, dpi=300, bbox_inches="tight")
    elif fig is not None:
        plt.show()

    return (fig or ax.figure), ax


def plt_dboundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "Decision Boundary",
    figsize: Tuple[int, int] = (6, 4),
    save_path: Optional[str] = None,
    cmap_continuous: str = "coolwarm",
    cmap_discrete: str = "Set3",
    ax: Optional[plt.Axes] = None,
):

    plt.rcParams["font.family"] = "DejaVu Sans"
    
    xx, yy, grid = _mesh_2d(X)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if _is_continuous_model(model):
        Z = _continuous_score(model, grid).reshape(xx.shape)
        cs = ax.contourf(xx, yy, Z, levels=50,
                         alpha=0.25, cmap=cmap_continuous)

        (fig or ax.figure).colorbar(cs, ax=ax,
                                    pad=0.02).set_label("P(class=1) / score")

        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    else:
        Z = model.predict(grid).astype(float).reshape(xx.shape)
        levels = np.arange(-0.5, Z.max() + 1.5, 1)
        ax.contourf(xx, yy, Z, levels=levels, alpha=0.25, cmap=cmap_discrete)
        ax.contour(xx, yy, Z, levels=np.arange(0.5, Z.max() + 0.5, 1), colors="black", linewidths=1)

    classes = np.unique(y)
    colors = _class_palette(len(classes))

    for i, cls in enumerate(classes):
        m = (y == cls)
        ax.scatter(X[m, 0], X[m, 1], s=40, color=colors[i], label=f"Class {cls}")

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        (fig or ax.figure).savefig(save_path, dpi=300, bbox_inches="tight")

    elif fig is not None:
        plt.show()

    return (fig or ax.figure), ax


def plt_clusters(
    X: np.ndarray,
    y_pred: np.ndarray,
    model,
    *,
    y_true: Optional[np.ndarray] = None,
    title: str = "K-means (Clusters + Boundary)",
    alpha: float = 0.9,
    figsize: Tuple[int, int] = (6, 4),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):

    clusters = np.unique(y_pred)
    colors = plt.cm.Dark2(np.linspace(0, 1, len(clusters)))
    markers = ["*", "^", "s", "P", "X", "D"]

    xx, yy, grid = _mesh_2d(X)
    Z = model.predict(grid).reshape(xx.shape)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    levels = np.arange(-0.5, Z.max() + 1.5, 1)
    ax.contourf(xx, yy, Z, levels=levels, alpha=0.18, cmap="Set1")
    # ax.contour(xx, yy, Z, levels=np.arange(0.5, Z.max() + 0.5, 1), colors="black", linewidths=1)
    ax.contour(xx, yy, Z, levels=np.arange(0.5, Z.max()+0.5, 1), 
               colors="black", linewidths=1.5, linestyles="--")

    for i, c in enumerate(clusters):
        m = (y_pred == c)
        ax.scatter(X[m, 0], X[m, 1], s=70, alpha=alpha, color=colors[i], marker=markers[i % len(markers)], label=f"Cluster {c}")

    if hasattr(model, "cluster_centers_"):
        C = model.cluster_centers_
        ax.scatter(C[:, 0], C[:, 1], c="black", s=140, marker="x", linewidths=2, label="Centroids")

    if y_true is not None:
        y_map = _map_clusters_to_labels(y_true, y_pred)
        mis = (y_true != y_map)
        if np.any(mis):
            ax.scatter(X[mis, 0], X[mis, 1], facecolors="none", edgecolors="red", s=130, linewidths=1.2, label="Misclassified")

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        (fig or ax.figure).savefig(save_path, dpi=300, bbox_inches="tight")
    elif fig is not None:
        plt.show()

    return (fig or ax.figure), ax


def plt_fcm(
    X: np.ndarray,
    u_class1: np.ndarray,
    *,
    title: str = "FCM: membership to class 1",
    figsize: Tuple[int, int] = (7, 5),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):

    u = np.clip(np.asarray(u_class1, dtype=float), 0.0, 1.0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(X[:, 0], X[:, 1], c=u, s=40, cmap="viridis", vmin=0.0, vmax=1.0)
    (fig or ax.figure).colorbar(sc, ax=ax, pad=0.02).set_label("membership of class 1")

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        (fig or ax.figure).savefig(save_path, dpi=300, bbox_inches="tight")
    elif fig is not None:
        plt.show()

    return (fig or ax.figure), ax


def plt_cmatrix(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    *,
    cm: Optional[np.ndarray] = None,
    labels: Optional[Iterable[str]] = None,
    normalize: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (4, 3),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):

    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Provide either cm=... or y_true and y_pred.")
        cm = sk_confusion_matrix(y_true, y_pred)

    cm = np.asarray(cm)

    if labels is None:
        n = cm.shape[0]
        labels = [f"Class {i}" for i in range(n)]

    if normalize is not None:

        with np.errstate(all="ignore"):
            if normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == "all":
                cm = cm / cm.sum()
            else:
                raise ValueError('normalize must be None, "true", "pred", or "all"')

        cm = np.nan_to_num(cm)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()

    if save_path:
        (fig or ax.figure).savefig(save_path, dpi=300, bbox_inches="tight")

    elif fig is not None:
        plt.show()

    return (fig or ax.figure), ax


def plt_dtree(
    model,
    *,
    feature_names=("x1", "x2"),
    class_names=("Class 0", "Class 1"),
    max_depth=None,
    figsize=(5, 3),
    dpi=160,
    filled=True,
    rounded=True,
    impurity=False,
    proportion=True,
    precision=2,
    style="seaborn-v0_8-whitegrid",
    title="• Decision Tree",
    save_path=None,
):

    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass

    est = getattr(model, "named_steps", None)
    clf = est["tree"] if isinstance(est, dict) and "tree" in est else (
        list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    plot_tree(
        clf,
        feature_names=list(feature_names),
        class_names=list(class_names),
        max_depth=max_depth,
        filled=filled,       # usa o mapa de cores da árvore
        rounded=rounded,
        impurity=impurity,
        proportion=proportion,
        precision=precision,
        fontsize=5,
        ax=ax,
    )

    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=7, pad=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    return fig, ax


def export_tree_text(model, *, feature_names=("x1", "x2")) -> str:
    """
    Return a plain-text representation of the trained decision tree rules.
    """

    est = getattr(model, "named_steps", None)
    clf = est["tree"] if isinstance(est, dict) and "tree" in est else model

    return export_text(clf, feature_names=list(feature_names))

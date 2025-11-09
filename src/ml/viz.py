import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def scatter_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "Dataset Scatter",
    show_ellipses: bool = True,
    alpha: float = 0.85,
    figsize: tuple[int, int] = (7, 5),
    save_path: str | None = None,
):
    """
    Scatter plot for labeled datasets with optional Gaussian ellipses.
    """

    classes = np.unique(y)
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=figsize)

    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   s=40, alpha=alpha,
                   color=colors[i],
                   label=f"Class {cls}")

        if show_ellipses:
            mx, my = X[mask, 0].mean(), X[mask, 1].mean()
            sx, sy = X[mask, 0].std(), X[mask, 1].std()
            t = np.linspace(0, 2 * np.pi, 200)
            xe = mx + sx * np.cos(t)
            ye = my + sy * np.sin(t)
            ax.plot(xe, ye, color=colors[i], lw=2)

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax


def scatter_clusters(
    X, y_pred, model, *,
    title="K-means Clusters + Boundaries",
    alpha=0.85, figsize=(7, 5), save_path=None
):
    clusters = np.unique(y_pred)
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    pad = 1.0

    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, Z, alpha=0.15, levels=len(clusters), cmap=plt.cm.tab10)

    for i, c in enumerate(clusters):
        m = y_pred == c
        ax.scatter(X[m, 0], X[m, 1], s=40, alpha=alpha, color=colors[i], label=f"Cluster {c}")

    if hasattr(model, "cluster_centers_"):
        C = model.cluster_centers_
        ax.scatter(C[:, 0], C[:, 1], c="black", s=120, marker="x", linewidths=2, label="Centroids")

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax


def scatter_cm(cm: np.ndarray,
               labels=None,
               title="Confusion Matrix",
               save_path=None):

    """
    Pretty confusion matrix visualization.
    """

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=labels or ["Class 0", "Class 1"],
                yticklabels=labels or ["Class 0", "Class 1"],
                ax=ax)

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax


def scatter_fcm(
    X: np.ndarray,
    u_class1: np.ndarray,
    *,
    title: str = "Fuzzy membership to Class 1",
    figsize: tuple[int, int] = (7, 5),
    save_path: str | None = None,
):
    """
    Color points by membership degree to class 1 (values in [0,1]).
    """
    u = np.clip(np.asarray(u_class1, dtype=float), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(X[:, 0], X[:, 1], c=u, s=40, cmap="viridis", vmin=0.0, vmax=1.0)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("membership of class 1")

    ax.set_title(title, fontsize=14, loc="left")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax

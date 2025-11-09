import numpy as np

from typing import Dict, Any


def _init_membership(n_samples: int,
                     c: int,
                     rng: np.random.Generator) -> np.ndarray:

    U = rng.random((c, n_samples))
    U /= U.sum(axis=0, keepdims=True)

    return U


def fcm_fit(
    X: np.ndarray,
    c: int = 2,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int = 42,
) -> Dict[str, Any]:

    """
    Basic Fuzzy C-means implementation.

    Returns
    -------
    {
      "centers": np.ndarray (c, d),
      "U": np.ndarray (c, n),            # memberships, columns sum to 1
      "labels": np.ndarray (n,),         # hard labels = argmax(U, axis=0)
      "n_iter": int
    }
    """

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(int(seed))

    U = _init_membership(n, c, rng)
    um = U ** m

    for it in range(1, max_iter + 1):
        # update centers
        num = um @ X                       # (c,d)
        den = um.sum(axis=1, keepdims=True)  # (c,1)
        centers = num / np.maximum(den, 1e-12)

        # distances to centers
        # D[j, i] = ||x_i - v_j||_2
        diff = X[None, :, :] - centers[:, None, :]         # (c,n,d)
        D = np.linalg.norm(diff, axis=2) + 1e-12           # (c,n) avoid zeros

        # update U
        power = 2.0 / (m - 1.0)
        inv = (D[:, None, :] / D[None, :, :]) ** power     # (c,c,n)
        U_new = 1.0 / inv.sum(axis=1)                      # (c,n)

        # convergence
        delta = np.linalg.norm(U_new - U)
        U, um = U_new, U_new ** m

        if delta < tol:
            break

    labels = U.argmax(axis=0)

    return {"centers": centers, "U": U, "labels": labels, "n_iter": it}


def run_fcm(
    X: np.ndarray,
    c: int = 2,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int = 42,
) -> Dict[str, Any]:

    """
    Convenience wrapper to run FCM with typical defaults.
    """

    return fcm_fit(X, c=c, m=m, max_iter=max_iter, tol=tol, seed=seed)

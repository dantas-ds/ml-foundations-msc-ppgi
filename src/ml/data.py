import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Protocol


class DatasetGenerator(Protocol):
    """
    Strategy interface for dataset generators.
    """

    def generate(self,
                 params: Dict[str, Any],
                 seed: int) -> Tuple[np.ndarray, np.ndarray]: ...


class BivariateGaussianGenerator:
    """
    Generate two-class 2D Gaussian blobs with diagonal covariance.
    """

    def generate(self, params: Dict[str, Any],
                 seed: int) -> Tuple[np.ndarray, np.ndarray]:

        rng = np.random.default_rng(int(seed))

        n1 = int(params["n_class1"])
        n2 = int(params["n_class2"])

        mu1 = np.asarray(params["mu1"], dtype=float)
        mu2 = np.asarray(params["mu2"], dtype=float)

        std1 = np.sqrt(np.asarray(params["var1"], dtype=float))
        std2 = np.sqrt(np.asarray(params["var2"], dtype=float))

        x1 = rng.normal(loc=mu1, scale=std1, size=(n1, 2))
        x2 = rng.normal(loc=mu2, scale=std2, size=(n2, 2))

        X = np.vstack([x1, x2]).astype(np.float64)
        y = np.concatenate([np.zeros(n1, dtype=np.int64),
                            np.ones(n2, dtype=np.int64)])

        return X, y


GENERATORS: Dict[str, DatasetGenerator] = {
    "bivariate_gaussian": BivariateGaussianGenerator(),
}


def generate_data(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatch dataset generation using a registry of strategies.
    """

    ds = cfg["dataset"]
    gen_name = ds.get("generator", "")
    params = ds.get("params", {})
    seed = int(params.get("random_state", cfg.get("seed", 42)))

    if gen_name not in GENERATORS:
        raise ValueError(f"Unsupported generator: {gen_name}")

    return GENERATORS[gen_name].generate(params, seed)


def save_dataset(X: np.ndarray,
                 y: np.ndarray,
                 path: str | Path,
                 meta: Optional[Dict[str, Any]] = None) -> Path:
    """
    Persist arrays as compressed NPZ; optional small metadata dict.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if meta is None:
        np.savez_compressed(p, X=X, Y=y)

    else:
        np.savez_compressed(p, X=X, Y=y, META=np.array([meta], dtype=object))

    return p


def load_dataset(path: str | Path) -> Tuple[np.ndarray,
                                            np.ndarray,
                                            Optional[Dict[str, Any]]]:

    """
    Load NPZ dataset with optional metadata.
    """

    with np.load(path, allow_pickle=True) as f:
        X = f["X"]
        Y = f["Y"]

        meta = None

        if "META" in f:
            arr = f["META"]

            if arr.size > 0:
                meta = arr[0].item()

    return X, Y, meta


def export_csv(X: np.ndarray,
               y: np.ndarray,
               path: str | Path) -> Path:

    """
    Export as CSV with header x1,x2,y for inspection.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([X, y.astype(np.int64)])

    np.savetxt(p, data,
               delimiter=",",
               fmt=["%.10f", "%.10f", "%d"],
               header="x1,x2,y",
               comments="")

    return p

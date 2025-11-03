import numpy as np


def generate_data(cfg: dict):
    ds = cfg["dataset"]
    gen = ds.get("generator", "")
    p = ds.get("params", {})
    seed = int(p.get("random_state", 42))

    if gen == "bivariate_gaussian":
        return gen_bivariate_gaussian(p, seed)

    raise ValueError(f"Unsupported generator: {gen}")


def gen_bivariate_gaussian(p: dict, seed: int = 42):
    rng = np.random.default_rng(seed)
    print(p['n_class1'], p['n_class2'])

    n1, n2 = int(p["n_class1"]), int(p["n_class2"])
    mu1 = np.array(p["mu1"])
    mu2 = np.array(p["mu2"])

    std1 = np.sqrt(np.array(p["var1"]))
    std2 = np.sqrt(np.array(p["var2"]))

    x1 = rng.normal(loc=mu1, scale=std1, size=(n1, 2))
    x2 = rng.normal(loc=mu2, scale=std2, size=(n2, 2))

    X = np.vstack([x1, x2])
    y = np.concatenate([np.zeros(n1, int), np.ones(n2, int)])

    return X, y

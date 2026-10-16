"""Equal-weight clip means and the paper's percentile bootstrap intervals."""
import numpy as np

BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10000


def summarize(values, rng=None, indices=None):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    assert np.isfinite(values).all(), "Nonfinite metric"
    if not len(values):
        return {"n": 0, "mean": None, "ci95": None}
    if indices is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED) if rng is None else rng
        indices = rng.integers(len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return {"n": len(values), "mean": float(values.mean()), "ci95": [float(low), float(high)]}

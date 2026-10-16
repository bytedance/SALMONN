"""Optional listening pose FID from the historical exploratory supplement.

Features are 3D head axis-angle rotations relative to each clip's first frame.
Pool sufficient statistics across clips; averaging per-clip FIDs is different.
"""
import numpy as np
from scipy.linalg import sqrtm


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "sum": values.sum(0).tolist(),
            "outer": (values.T @ values).tolist()}


def merge(rows):
    count = sum(row["n"] for row in rows)
    assert count > 1, "Pose FID needs at least two valid frames"
    total = sum(np.array(row["sum"]) for row in rows)
    outer = sum(np.array(row["outer"]) for row in rows)
    # Unbiased pooled covariance, preserving the original aggregation order.
    return total / count, (outer - np.outer(total, total) / count) / (count - 1)


def frechet_distance(predicted, reference):
    mean, covariance = merge(predicted)
    ref_mean, ref_covariance = merge(reference)
    root = sqrtm(covariance @ ref_covariance)
    if np.iscomplexobj(root):
        assert np.max(np.abs(root.imag)) < 1e-8
    value = float((mean - ref_mean) @ (mean - ref_mean)
                  + np.trace(covariance + ref_covariance) - 2 * np.trace(root.real))
    assert value > -1e-8, value
    return max(value, 0.)

"""Utilities for clustering numerical material properties."""

from numbers import Integral

import numpy as np


def cluster_materials(materials, frequencies, count):
    """Return weighted material centers and an index for each input row.

    Each row holds the numerical properties of one material. Frequencies
    weight the fit, and distances are measured in the units of the supplied
    properties. The result contains at most count centers. Use the returned
    indices to remap existing material references.
    """
    rows = np.asarray(materials, dtype=np.float64)
    weights = np.asarray(frequencies)
    if rows.ndim != 2 or not all(rows.shape) or not np.isfinite(rows).all():
        raise ValueError("materials must be a nonempty table of finite numerical rows")
    if (weights.shape != (len(rows),) or weights.dtype.kind not in "biuf"
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.any(weights > 0)):
        raise ValueError("frequencies must be finite and nonnegative, with at least one positive weight")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    if len(rows) <= count:
        return rows.copy(), np.arange(len(rows))

    from sklearn.cluster import KMeans

    count = min(int(count), np.count_nonzero(weights))
    model = KMeans(n_clusters=count, n_init=3, random_state=21)
    model.fit(rows, sample_weight=weights)
    used, remap = np.unique(model.labels_, return_inverse=True)
    return model.cluster_centers_[used], remap

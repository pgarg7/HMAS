"""Figure 14 -- HMAS dimension cross-correlation heatmap.

Regenerates the 6x6 Pearson-correlation heatmap of HMAS sub-metrics from an
``hmas_correlation.json`` artifact (containing ``correlation_matrix`` and
``metric_names``), or recomputes it from a per-model scores dict.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import style
from ._io import DataLike, load


def _matrix_from_scores(scores: dict):
    """Build the 6xN matrix and correlation from {model: {sub-metric: val}}."""
    keys = style.METRIC_KEYS
    rows = []
    for _, s in scores.items():
        row = [s.get(k, np.nan) for k in keys]
        rows.append(row)
    mat = np.asarray(rows, dtype=float)
    complete = np.isfinite(mat).all(axis=1)
    mat = mat[complete]
    corr = np.corrcoef(mat.T) if mat.shape[0] >= 3 else np.full(
        (len(keys), len(keys)), np.nan)
    return corr, int(mat.shape[0])


def hmas_correlation(data: DataLike,
                     eval_day: int = 5,
                     ax=None,
                     title: Optional[str] = None):
    """Draw the HMAS sub-metric correlation heatmap.

    ``data`` may be (a) an ``hmas_correlation.json`` dict/path with a
    precomputed ``correlation_matrix``, or (b) a ``{model: {sub-metric: val}}``
    scores dict, in which case the correlation is computed here. Returns the
    Figure.
    """
    import matplotlib.pyplot as plt

    d = load(data)
    if isinstance(d, dict) and "correlation_matrix" in d:
        corr = np.asarray(d["correlation_matrix"], dtype=float)
        n_complete = int(d.get("n_models_complete", 0))
    else:
        corr, n_complete = _matrix_from_scores(d)

    labels = style.METRIC_LABELS
    n = corr.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels[:n], fontsize=12, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[:n], fontsize=12)

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    if title is None:
        title = (f"Cross-correlation of HMAS sub-metrics at Day {eval_day}\n"
                 f"({n_complete} models with complete data). "
                 "Low |r| confirms independent dimensions.")
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig

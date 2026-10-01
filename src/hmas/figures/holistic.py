"""Figure 11 -- Holistic Model Assessment Score radar chart.

Regenerates the per-model HMAS radar from a ``holistic_scores_day{N}.json``
artifact (or an equivalent dict): ``{model: {sfi, eff_res, tau_d, ees, pcs,
asi, HMAS}}``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import style
from ._io import DataLike, load

_EVAL_LABEL = {3: "synoptic", 5: "medium", 7: "medium", 10: "extended",
               15: "extended"}


def holistic_radar(scores: DataLike,
                   eval_day: int = 5,
                   models: Optional[Sequence[str]] = None,
                   ax=None,
                   title: Optional[str] = None):
    """Draw the HMAS radar. ``scores`` is the day-N holistic scores dict or a
    path to its JSON. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    data = load(scores)
    keys = style.order_models(models if models is not None else data.keys())
    keys = [k for k in keys if k in data]

    metric_keys = style.METRIC_KEYS
    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10),
                               subplot_kw=dict(polar=True))
    else:
        fig = ax.figure

    for key in keys:
        s = data[key]
        values = [s.get(m, 0.0) for m in metric_keys]
        values = [v if (v is not None and np.isfinite(v)) else 0.0
                  for v in values]
        values = values + [values[0]]
        ax.plot(angles, values, style.LINESTYLES.get(key, "-"),
                lw=1.8, color=style.model_color(key),
                label=style.model_label(key))
        ax.fill(angles, values, color=style.model_color(key), alpha=0.04)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(style.METRIC_LABELS, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    if title is None:
        label = _EVAL_LABEL.get(eval_day, "")
        rng = f" ({label}-range)" if label else ""
        title = (f"Holistic Model Assessment Score (HMAS) -- "
                 f"Day {eval_day}{rng}")
    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    return fig


def holistic_parallel(scores: DataLike,
                      eval_day: int = 5,
                      models: Optional[Sequence[str]] = None,
                      ax=None):
    """Parallel-coordinates companion to the radar (one line per model across
    the six sub-metric axes)."""
    import matplotlib.pyplot as plt

    data = load(scores)
    keys = style.order_models(models if models is not None else data.keys())
    keys = [k for k in keys if k in data]
    metric_keys = style.METRIC_KEYS
    x = np.arange(len(metric_keys))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    else:
        fig = ax.figure

    for key in keys:
        s = data[key]
        values = [s.get(m, np.nan) for m in metric_keys]
        ax.plot(x, values, style.LINESTYLES.get(key, "-"),
                marker="o", lw=1.8, color=style.model_color(key),
                label=style.model_label(key))

    ax.set_xticks(x)
    ax.set_xticklabels(style.METRIC_LABELS, fontsize=14)
    ax.set_ylabel("Score (0 = worst, 1 = best)", fontsize=14)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=5, fontsize=9)
    fig.suptitle(f"HMAS sub-metrics -- Day {eval_day}", fontsize=15)
    fig.tight_layout()
    return fig

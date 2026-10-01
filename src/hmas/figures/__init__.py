"""hmas.figures -- regenerate the paper's figures from saved metric artifacts.

Most figure functions accept either a path to the corresponding metrics JSON
(the artifacts shipped with the paper) or an already-loaded dict, and return a
matplotlib ``Figure``. Use :func:`hmas.figures.style.savefig` to write a
publication PDF/PNG pair.

Two artifact tiers:
  * Scalar metric JSONs (committed): drive ACC (Fig 4), HMAS radar (Fig 11),
    correlation (Fig 14), and the remaining summary figures.
  * Curve artifacts (produced once via :mod:`hmas.figures.curves` from the
    precomputed cache): drive the power-spectrum and RMSE-vs-lead figures
    (Figs 1-3), whose underlying curves are not in the scalar JSONs.

Implemented:
    spectral_comparison                 (Fig 1)   <- spectral_curves.json
    spectral_ratio                      (Fig 2)   <- spectral_ratio_curves.json
    rmse_vs_leadtime                    (Fig 3)   <- rmse_curves.json
    acc_vs_leadtime                     (Fig 4)   <- acc_metrics.json
    holistic_radar, holistic_parallel   (Fig 11)  <- holistic_scores_day{N}.json
    hmas_correlation                    (Fig 14)  <- hmas_correlation.json

In progress (figure-number order): Figs 5-10, 12-13.
"""
from __future__ import annotations

from . import curves, generate, style
from .correlation import hmas_correlation
from .holistic import holistic_parallel, holistic_radar
from .skill import acc_vs_leadtime, rmse_vs_leadtime
from .spectral import spectral_comparison, spectral_ratio
from .generate import generate_all, write_hmas_artifacts

__all__ = [
    "style", "curves", "generate",
    "generate_all", "write_hmas_artifacts",
    "spectral_comparison",
    "spectral_ratio",
    "rmse_vs_leadtime",
    "acc_vs_leadtime",
    "holistic_radar",
    "holistic_parallel",
    "hmas_correlation",
]

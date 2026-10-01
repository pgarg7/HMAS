"""Directory-driven figure generation.

The intended workflow:

1. Run the metric computation (HMAS + the other diagnostics) and save every
   artifact into one results directory using the canonical filenames below
   (:func:`write_hmas_artifacts` does this for a package HMAS run).
2. Point :func:`generate_all` at that directory. It renders every figure whose
   input artifact is present and skips the rest, so reviewers reproduce the
   figures with nothing but the saved JSONs.

Canonical input filenames (one per figure input):
    spectral_curves.json          -> Fig 1
    spectral_ratio_curves.json    -> Fig 2 (u500, z500)
    rmse_curves.json              -> Fig 3
    acc_metrics.json              -> Fig 4
    holistic_scores_day{N}.json   -> Fig 11 (radar + parallel), per day
    hmas_correlation.json         -> Fig 14
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import style
from ._io import load
from .correlation import hmas_correlation
from .holistic import holistic_parallel, holistic_radar
from .skill import acc_vs_leadtime, rmse_vs_leadtime
from .spectral import spectral_comparison, spectral_ratio

# output_stem -> (input filename, render callable taking the loaded data)
# A trailing "{day}" in the stem is expanded over the days found on disk.
FIGURE_REGISTRY: Dict[str, Tuple[str, Callable]] = {
    "fig01_spectral_comparison":   ("spectral_curves.json",
                                    spectral_comparison),
    "fig02_spectral_ratio_u500":   ("spectral_ratio_curves.json",
                                    lambda d: spectral_ratio(d, "u500")),
    "fig02_spectral_ratio_z500":   ("spectral_ratio_curves.json",
                                    lambda d: spectral_ratio(d, "z500")),
    "fig03_rmse_vs_leadtime":      ("rmse_curves.json", rmse_vs_leadtime),
    "fig04_acc_vs_leadtime":       ("acc_metrics.json", acc_vs_leadtime),
    "fig14_hmas_correlation_day5": ("hmas_correlation.json", hmas_correlation),
}

# Per-day figures: stem template -> (filename template, render(data, day)).
PER_DAY_REGISTRY: Dict[str, Tuple[str, Callable]] = {
    "fig11_holistic_radar_day{day}": ("holistic_scores_day{day}.json",
                                      lambda d, day: holistic_radar(d, day)),
    "fig11_holistic_parallel_day{day}": ("holistic_scores_day{day}.json",
                                         lambda d, day: holistic_parallel(d, day)),
}
DEFAULT_DAYS = (3, 5, 15)


def generate_all(results_dir: str,
                 out_dir: Optional[str] = None,
                 days: Sequence[int] = DEFAULT_DAYS,
                 figures: Optional[Sequence[str]] = None,
                 dpi: int = 300,
                 also_png: bool = True,
                 verbose: bool = True) -> List[str]:
    """Render every figure whose input artifact exists in ``results_dir``.

    Parameters
    ----------
    results_dir:
        Directory holding the metric JSON artifacts (canonical filenames).
    out_dir:
        Where to write the figures (defaults to ``results_dir``).
    days:
        Horizons for the per-day figures (radar/parallel).
    figures:
        Optional whitelist of output stems to render; default is all.
    Returns the list of written file stems.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    out_dir = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    def _emit(stem, fname, render):
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            if verbose:
                print(f"  skip {stem}: missing {fname}")
            return
        try:
            fig = render(load(path))
        except Exception as e:           # noqa: keep one figure's failure local
            if verbose:
                print(f"  FAIL {stem}: {type(e).__name__}: {e}")
            return
        stem_path = os.path.join(out_dir, stem)
        style.savefig(fig, stem_path, dpi=dpi, also_png=also_png)
        plt.close(fig)
        written.append(stem_path)
        if verbose:
            print(f"  wrote {stem}")

    for stem, (fname, render) in FIGURE_REGISTRY.items():
        if figures is None or stem in figures:
            _emit(stem, fname, render)

    for tmpl, (fname_tmpl, render) in PER_DAY_REGISTRY.items():
        for day in days:
            stem = tmpl.format(day=day)
            if figures is not None and stem not in figures:
                continue
            _emit(stem, fname_tmpl.format(day=day),
                  (lambda r, _d=day, _f=render: _f(r, _d)))

    return written


# ---------------------------------------------------------------------------
# Writer: emit canonical artifacts from a package HMAS run so the figures above
# can consume them without the original analysis script.
# ---------------------------------------------------------------------------
def write_hmas_artifacts(result, results_dir: str,
                         correlation_day: int = 5) -> List[str]:
    """Write ``holistic_scores_day{N}.json`` for each eval day and
    ``hmas_correlation.json`` from an :class:`hmas.HMASResult`. Returns the
    written paths."""
    os.makedirs(results_dir, exist_ok=True)
    paths = []
    metric_keys = style.METRIC_KEYS

    for day in result.eval_days:
        scores = {m: {k: s.get(k) for k in metric_keys + ["HMAS"]}
                  for m, s in result.scores(day).items()}
        p = os.path.join(results_dir, f"holistic_scores_day{day}.json")
        with open(p, "w") as f:
            json.dump(scores, f, indent=2)
        paths.append(p)

    # correlation from the chosen day's per-model sub-metrics
    day = correlation_day if correlation_day in result.eval_days \
        else result.eval_days[-1]
    scores = result.scores(day)
    rows, models = [], []
    for m, s in scores.items():
        row = [s.get(k, np.nan) for k in metric_keys]
        if all(v is not None and np.isfinite(v) for v in row):
            rows.append(row)
            models.append(m)
    if len(rows) >= 3:
        mat = np.asarray(rows, dtype=float)
        corr = np.corrcoef(mat.T)
        offdiag = ~np.eye(len(metric_keys), dtype=bool)
        log = {
            "correlation_matrix": corr.tolist(),
            "metric_names": metric_keys,
            "n_models_complete": len(models),
            "model_scores": {m: scores[m] for m in models},
            "mean_abs_offdiag_correlation":
                float(np.mean(np.abs(corr[offdiag]))),
        }
        p = os.path.join(results_dir, "hmas_correlation.json")
        with open(p, "w") as f:
            json.dump(log, f, indent=2)
        paths.append(p)
    return paths

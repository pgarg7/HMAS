"""
Error Consensus Ratio (ECR): a cross-model diagnostic.

Unlike the HMAS sub-metrics, ECR is not a per-model score; it partitions the
forecast error of an *ensemble of distinct models* into a shared component
(common to all, i.e. atmospheric-predictability-limited) and a
model-specific component. A high ECR means models err in the same way --
evidence that the remaining error is dominated by intrinsic predictability
rather than by modelling choices.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .stats import align_grids
from .stores import ForecastStore, VerificationSource


def _align_cube(cube_a: np.ndarray, cube_b: np.ndarray):
    """Align two ``(n, lat, lon)`` cubes on their common lat/lon extent."""
    nlat = min(cube_a.shape[1], cube_b.shape[1])
    nlon = min(cube_a.shape[2], cube_b.shape[2])
    return cube_a[:, :nlat, :nlon], cube_b[:, :nlat, :nlon]


def error_consensus_ratio(store: ForecastStore,
                          verification: VerificationSource,
                          models: Sequence[str],
                          init_time: str,
                          variable: str,
                          min_models: int = 3) -> Optional[Dict]:
    r"""ECR as a function of lead time for one init and variable.

    .. math::
        \mathrm{ECR}(\tau) = \frac{\langle\|\bar e\|^2\rangle}
            {\langle\|\bar e\|^2\rangle
             + \tfrac1M \sum_m \langle\|e_m - \bar e\|^2\rangle}

    where :math:`e_m` is model :math:`m`'s error field, :math:`\bar e` is the
    multi-model mean error, and :math:`\langle\cdot\rangle` is a spatial mean.

    Returns ``None`` if fewer than ``min_models`` models are available.
    Otherwise a dict with ``lead_hours``, ``ecr`` (per lead), ``n_models`` and
    ``model_keys``.
    """
    lh_truth, truth = verification.get_truth(variable, init_time)
    if lh_truth is None:
        return None

    model_cubes: Dict[str, np.ndarray] = {}
    for m in models:
        lh, cube = store.get_series(m, init_time, variable)
        if lh is None:
            continue
        n = min(len(lh), truth.shape[0])
        fa, _ = _align_cube(cube[:n], truth[:n])
        model_cubes[m] = fa

    avail = sorted(model_cubes.keys())
    if len(avail) < min_models:
        return None

    min_nlat = min(model_cubes[k].shape[1] for k in avail)
    min_nlon = min(model_cubes[k].shape[2] for k in avail)
    nsteps = min(model_cubes[k].shape[0] for k in avail)
    truth_sub = truth[:nsteps, :min_nlat, :min_nlon]

    M = len(avail)
    errors = np.stack([
        model_cubes[k][:nsteps, :min_nlat, :min_nlon] - truth_sub
        for k in avail], axis=0)  # (M, nsteps, nlat, nlon)

    e_bar = errors.mean(axis=0)
    shared_var = np.mean(e_bar ** 2, axis=(1, 2))
    specific_var = np.mean(
        [np.mean((errors[m] - e_bar) ** 2, axis=(1, 2)) for m in range(M)],
        axis=0)
    denom = shared_var + specific_var
    ecr_vals = np.where(denom > 0, shared_var / denom, np.nan)

    return {
        "lead_hours": lh_truth[:nsteps].astype(float),
        "ecr": ecr_vals,
        "n_models": M,
        "model_keys": avail,
    }


def pairwise_error_correlation(store: ForecastStore,
                               verification: VerificationSource,
                               models: Sequence[str],
                               init_time: str,
                               variable: str,
                               lead_hours: float,
                               tolerance: float = 12.0
                               ) -> Optional[Dict]:
    """Spatial correlation matrix of model error fields at one lead time.

    Returns ``{"model_keys": [...], "corr": (M, M) ndarray}`` or ``None``.
    High off-diagonal correlations corroborate a high ECR.
    """
    fields: Dict[str, np.ndarray] = {}
    for m in models:
        truth = verification.get_truth(variable, init_time)
        lh_t, cube_t = truth
        if lh_t is None:
            return None
        f = store.get_field(m, init_time, variable, lead_hours, tolerance)
        if f is None:
            continue
        idx = int(np.argmin(np.abs(lh_t - lead_hours)))
        if np.abs(lh_t[idx] - lead_hours) > tolerance:
            continue
        fa, ta = align_grids(f, cube_t[idx])
        fields[m] = (fa - ta).ravel()

    keys = sorted(fields.keys())
    if len(keys) < 2:
        return None
    M = len(keys)
    corr = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(i, M):
            ei, ej = fields[keys[i]], fields[keys[j]]
            valid = np.isfinite(ei) & np.isfinite(ej)
            if valid.sum() > 100:
                r = np.corrcoef(ei[valid], ej[valid])[0, 1]
                corr[i, j] = corr[j, i] = r
    return {"model_keys": keys, "corr": corr}

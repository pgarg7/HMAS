"""
Grid-handling and verification statistics (area-weighted RMSE, ACC).

All functions operate on plain ``numpy`` arrays and assume a regular
latitude-longitude grid ordered from North (+90) to South (-90) unless an
explicit latitude vector is supplied. This matches the convention used
throughout the diagnostic framework.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def align_grids(forecast: np.ndarray,
                analysis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align two 2-D fields by truncating the larger latitude dimension.

    A common off-by-one mismatch is 720 vs 721 latitudes (pole inclusion);
    this drops the trailing row from the taller field so both share a grid.
    """
    if forecast.shape == analysis.shape:
        return forecast, analysis
    nlat = min(forecast.shape[0], analysis.shape[0])
    return forecast[:nlat, :], analysis[:nlat, :]


def make_lat_weights(nlat: int,
                     lats: Optional[Sequence[float]] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine-of-latitude area weights for a regular lat-lon grid.

    Parameters
    ----------
    nlat:
        Number of latitude rows.
    lats:
        Optional explicit latitude values (degrees). Defaults to
        ``linspace(90, -90, nlat)``.

    Returns
    -------
    weights, lats:
        ``weights`` are normalised so their mean is 1.
    """
    if lats is None:
        lats = np.linspace(90, -90, nlat)
    lats = np.asarray(lats, dtype=float)
    cos_lat = np.cos(np.deg2rad(lats))
    return cos_lat / cos_lat.mean(), lats


def compute_weighted_rmse(forecast: np.ndarray,
                          analysis: np.ndarray,
                          lat_weights: Optional[np.ndarray] = None) -> float:
    r"""Global area-weighted RMSE.

    .. math::
        \mathrm{RMSE} = \sqrt{\tfrac{1}{A}\int (f-a)^2 \cos\phi\,
        \mathrm{d}\phi\,\mathrm{d}\lambda}

    Grids are aligned automatically if shapes differ. If ``lat_weights`` is
    ``None`` (or stale after re-alignment) cosine weights are recomputed.
    """
    forecast, analysis = align_grids(forecast, analysis)
    diff_sq = (forecast - analysis) ** 2
    if lat_weights is None or diff_sq.shape[0] != len(lat_weights):
        lat_weights = make_lat_weights(diff_sq.shape[0])[0]
    return float(np.sqrt(np.average(diff_sq, axis=0,
                                    weights=lat_weights).mean()))


def compute_acc(forecast: np.ndarray,
                analysis: np.ndarray,
                climatology_mean: np.ndarray,
                lat_weights: Optional[np.ndarray] = None) -> float:
    r"""Anomaly Correlation Coefficient (the standard WMO skill metric).

    .. math::
        \mathrm{ACC} = \frac{\sum_w f' a'}
                            {\sqrt{\sum_w f'^2 \cdot \sum_w a'^2}}

    with anomalies :math:`f' = f - c`, :math:`a' = a - c` about the
    climatological mean :math:`c`, and area weights :math:`w = \cos\phi`.
    """
    forecast, analysis = align_grids(forecast, analysis)
    if climatology_mean.shape != forecast.shape:
        climatology_mean, _ = align_grids(climatology_mean, forecast)
    f_anom = forecast - climatology_mean
    a_anom = analysis - climatology_mean

    if lat_weights is not None and len(lat_weights) == forecast.shape[0]:
        w = lat_weights[:, np.newaxis]
    else:
        w = make_lat_weights(forecast.shape[0])[0][:, np.newaxis]

    num = np.sum(w * f_anom * a_anom)
    denom = np.sqrt(np.sum(w * f_anom ** 2) * np.sum(w * a_anom ** 2))
    if denom < 1e-15:
        return np.nan
    return float(num / denom)


def compute_pointwise_rmse(forecast_list: Sequence[np.ndarray],
                           analysis_list: Sequence[np.ndarray]
                           ) -> Optional[np.ndarray]:
    r"""Pointwise RMSE map across several cases (e.g. init dates)."""
    n = len(forecast_list)
    if n == 0:
        return None
    aligned = [align_grids(f, a) for f, a in zip(forecast_list, analysis_list)]
    sum_sq = np.zeros_like(aligned[0][0])
    for f, a in aligned:
        sum_sq += (f - a) ** 2
    return np.sqrt(sum_sq / n)


def compute_pointwise_bias(forecast_list: Sequence[np.ndarray],
                           analysis_list: Sequence[np.ndarray]
                           ) -> Optional[np.ndarray]:
    r"""Pointwise mean bias map across several cases."""
    n = len(forecast_list)
    if n == 0:
        return None
    aligned = [align_grids(f, a) for f, a in zip(forecast_list, analysis_list)]
    total = np.zeros_like(aligned[0][0])
    for f, a in aligned:
        total += (f - a)
    return total / n


def global_kinetic_energy(u_field: np.ndarray,
                          v_field: np.ndarray,
                          lats: Optional[Sequence[float]] = None) -> float:
    r"""Area-weighted global kinetic energy :math:`\tfrac12(u^2+v^2)`.

    Longitude is averaged first, then a cosine-of-latitude weighted average is
    taken over latitude to correct for meridian convergence on a regular
    lat-lon grid. Used to build the energy series consumed by ASI.
    """
    u_field, v_field = align_grids(u_field, v_field)
    ny = u_field.shape[0]
    if lats is None:
        lats = np.linspace(90, -90, ny)
    cos_lat = np.cos(np.deg2rad(np.asarray(lats, dtype=float)))
    ke = 0.5 * (u_field ** 2 + v_field ** 2)
    ke_zonal = ke.mean(axis=1)
    return float(np.average(ke_zonal, weights=cos_lat))

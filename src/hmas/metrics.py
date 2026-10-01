"""
The six HMAS sub-metrics and the composite Holistic Model Assessment Score.

Every sub-metric maps onto ``[0, 1]`` (1 = best). The functions are pure: they
take ``numpy`` fields / series and return floats, with no I/O or global state.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import config
from .spectral import (compute_power_spectrum_2d, compute_spectral_energy_ratio,
                       eff_res_from_spectra, sfi_from_spectra)
from .stats import align_grids


def spectral_fidelity_index(forecast_field: np.ndarray,
                            analysis_field: np.ndarray,
                            k_min: int = 1, k_max: int = 200) -> float:
    """Spectral Fidelity Index (SFI) computed from two raw fields.

    Convenience wrapper that FFTs both fields then calls
    :func:`hmas.spectral.sfi_from_spectra`. In batch pipelines, pre-compute
    spectra once and call ``sfi_from_spectra`` directly.
    """
    forecast_field, analysis_field = align_grids(forecast_field, analysis_field)
    k_f, p_f = compute_power_spectrum_2d(forecast_field)
    k_a, p_a = compute_power_spectrum_2d(analysis_field)
    return sfi_from_spectra(k_f, p_f, k_a, p_a, k_min=k_min, k_max=k_max)


def effective_resolution(forecast_field: np.ndarray,
                         analysis_field: np.ndarray,
                         threshold: float = 0.5,
                         excess_cap: float = 2.0,
                         norm: float = 300.0) -> float:
    """Effective resolution from two raw fields (see ``eff_res_from_spectra``)."""
    k, ratio = compute_spectral_energy_ratio(forecast_field, analysis_field)
    within = np.where((ratio >= threshold) & (ratio <= excess_cap)
                      & (k >= 1))[0]
    if len(within) == 0:
        return 0.0
    return min(1.0, float(k[within[-1]]) / norm)


def error_doubling_time(rmse_series: Sequence[float],
                        lead_hours: Sequence[float],
                        normalize_max_days: float = 2.0) -> float:
    r"""Normalised error-doubling time :math:`\tau_d`.

    Fits :math:`\log(\mathrm{RMSE}) = \lambda t + c` over the growth phase and
    sets :math:`\tau_d = \ln 2 / \lambda`, normalised by
    ``normalize_max_days`` (default 2 days, the approximate theoretical
    maximum for mid-latitude synoptic predictability; Lorenz 1969).

    Returns ``1.0`` if no positive growth rate is detected, ``nan`` if the
    series is too short to fit.
    """
    rmse = np.asarray(rmse_series, dtype=float)
    hours = np.asarray(lead_hours, dtype=float)
    if len(rmse) < 3 or np.all(rmse <= 0):
        return np.nan
    rmse_max = np.max(rmse)
    growth_mask = rmse < 0.8 * rmse_max
    if growth_mask.sum() < 3:
        growth_mask = np.ones(len(rmse), dtype=bool)
        growth_mask[0] = False
    rmse_g = rmse[growth_mask]
    hours_g = hours[growth_mask]
    pos = rmse_g > 0
    if pos.sum() < 2:
        return np.nan
    try:
        log_rmse = np.log(rmse_g[pos])
        lam = np.polyfit(hours_g[pos], log_rmse, 1)[0]
        if lam <= 0:
            return 1.0
        tau_d_hours = np.log(2) / lam
        return float(min(1.0, tau_d_hours / (normalize_max_days * 24)))
    except Exception:
        return np.nan


def extreme_event_skill(forecast_field: np.ndarray,
                        analysis_field: np.ndarray,
                        climatology_mean: np.ndarray,
                        climatology_std: np.ndarray,
                        threshold_sigma: float = 2.0
                        ) -> Tuple[float, float, float]:
    r"""Extreme Event Skill (EES) for warm-tail exceedances.

    .. math::
        \mathrm{EES} = \max\!\Big(0,\; 1 -
        \frac{\mathrm{RMSE}_\mathrm{cond}}{3\,\mathrm{RMSE}_\mathrm{uncond}}\Big)

    where ``RMSE_cond`` is over grid points whose *analysis* value exceeds
    :math:`\mu + \sigma_\mathrm{thresh}\,\sigma` (climatological mean and std).

    Returns ``(ees, conditional_bias, conditional_rmse)``. All ``nan`` if
    fewer than ten exceedance points exist.
    """
    forecast_field, analysis_field = align_grids(forecast_field, analysis_field)
    if climatology_mean.shape != forecast_field.shape:
        climatology_mean, _ = align_grids(climatology_mean, forecast_field)
    if climatology_std.shape != forecast_field.shape:
        climatology_std, _ = align_grids(climatology_std, forecast_field)

    threshold = climatology_mean + threshold_sigma * climatology_std
    extreme_mask = analysis_field > threshold
    if extreme_mask.sum() < 10:
        return np.nan, np.nan, np.nan

    diff = forecast_field[extreme_mask] - analysis_field[extreme_mask]
    cond_bias = float(np.mean(diff))
    cond_rmse = float(np.sqrt(np.mean(diff ** 2)))
    uncond_rmse = float(np.sqrt(np.mean(
        (forecast_field - analysis_field) ** 2)))
    if uncond_rmse == 0:
        return np.nan, cond_bias, cond_rmse
    ees = max(0.0, 1.0 - cond_rmse / (3.0 * uncond_rmse))
    return ees, cond_bias, cond_rmse


def physical_consistency_score(u_field: np.ndarray,
                               v_field: np.ndarray,
                               z_field: np.ndarray,
                               lat_grid: np.ndarray,
                               t_field: Optional[np.ndarray] = None,
                               balance_weights: Optional[Dict[str, float]] = None
                               ) -> Tuple[float, Dict[str, float]]:
    r"""Multi-balance Physical Consistency Score (PCS).

    Combines up to four independent dynamical-balance diagnostics, each mapped
    to ``[0, 1]`` and evaluated in the midlatitudes (|lat| in 20-70 deg):

    1. **Geostrophic balance** -- agreement of winds with the pressure field,
       :math:`v_g = f^{-1}\,\hat k \times \nabla\Phi`.
    2. **Non-divergence** -- ratio of RMS divergence to RMS vorticity, which
       should be small for balanced 500 hPa flow.
    3. **Thermal wind** (needs ``t_field``) -- correlation of meridional
       height and temperature gradients.
    4. **Hydrostatic consistency** (needs ``t_field``) -- correlation of
       geopotential height with layer-temperature-implied height.

    ``z_field`` units are auto-detected: a mean magnitude below 10000 is
    treated as geopotential *height* (m), otherwise as geopotential
    (m^2 s^-2). Returns ``(composite_pcs, sub_scores)``.
    """
    g = config.G
    omega = config.OMEGA
    R_earth = config.R_EARTH
    R_d = config.R_DRY
    ny, nx = z_field.shape
    lat_grid = np.asarray(lat_grid, dtype=float)

    z_mean = np.nanmean(np.abs(z_field))
    if z_mean < 10000:
        phi = z_field * g       # height -> geopotential
        z_height = z_field      # already metres
    else:
        phi = z_field           # already geopotential
        z_height = z_field / g

    lats_rad = np.deg2rad(lat_grid)
    f = 2 * omega * np.sin(lats_rad)
    dlat = np.abs(lat_grid[1] - lat_grid[0]) if ny > 1 else 0.25
    dy = np.deg2rad(dlat) * R_earth
    dlon = np.deg2rad(360.0 / nx) * R_earth

    dphi_dy = np.gradient(phi, -dy, axis=0)
    cos_lat = np.cos(lats_rad)
    dx_local = np.clip(dlon * cos_lat[:, np.newaxis], 1e3, None)
    dphi_dx = np.gradient(phi, axis=1) / dx_local
    f_2d = f[:, np.newaxis]
    f_safe = np.where(np.abs(f_2d) > 1e-6, f_2d, np.nan)
    u_g = -dphi_dy / f_safe
    v_g = dphi_dx / f_safe

    lat_mask = (np.abs(lat_grid) >= 20) & (np.abs(lat_grid) <= 70)
    if lat_mask.sum() == 0:
        return np.nan, {}

    u_mid, v_mid = u_field[lat_mask, :], v_field[lat_mask, :]
    ug_mid, vg_mid = u_g[lat_mask, :], v_g[lat_mask, :]
    valid = np.isfinite(ug_mid) & np.isfinite(vg_mid)
    if valid.sum() < 100:
        return np.nan, {}

    sub: Dict[str, float] = {}

    # 1. Geostrophic balance
    ageo_sq = ((u_mid[valid] - ug_mid[valid]) ** 2
               + (v_mid[valid] - vg_mid[valid]) ** 2)
    wind_sq = u_mid[valid] ** 2 + v_mid[valid] ** 2
    agr = np.sqrt(np.mean(ageo_sq) / max(np.mean(wind_sq), 1e-10))
    sub["geostrophic"] = float(max(0.0, 1.0 - agr / 0.5))

    # 2. Non-divergence
    du_dx = np.gradient(u_field, axis=1) / dx_local
    dv_dy = np.gradient(v_field, -dy, axis=0)
    dv_dx = np.gradient(v_field, axis=1) / dx_local
    du_dy = np.gradient(u_field, -dy, axis=0)
    div_mid = (du_dx + dv_dy)[lat_mask, :]
    vort_mid = (dv_dx - du_dy)[lat_mask, :]
    valid_dv = np.isfinite(div_mid) & np.isfinite(vort_mid)
    if valid_dv.sum() >= 100:
        rms_div = np.sqrt(np.mean(div_mid[valid_dv] ** 2))
        rms_vort = np.sqrt(np.mean(vort_mid[valid_dv] ** 2))
        ndr = rms_div / max(rms_vort, 1e-12)
        sub["nondivergence"] = float(max(0.0, 1.0 - ndr / 0.5))
    else:
        sub["nondivergence"] = np.nan

    # 3 & 4. Thermal wind + hydrostatic (require temperature)
    if t_field is not None and t_field.shape[0] >= ny:
        t_aligned = t_field[:ny, :nx]

        dz_dy_mid = np.gradient(z_height, -dy, axis=0)[lat_mask, :]
        dt_dy_mid = np.gradient(t_aligned, -dy, axis=0)[lat_mask, :]
        valid_tw = (np.isfinite(dz_dy_mid) & np.isfinite(dt_dy_mid)
                    & (np.abs(dz_dy_mid) > 1e-10))
        if valid_tw.sum() >= 100:
            r_tw = np.corrcoef(dz_dy_mid[valid_tw].ravel(),
                               dt_dy_mid[valid_tw].ravel())[0, 1]
            sub["thermal_wind"] = float(max(0.0, min(1.0, (r_tw + 0.2) / 1.2)))
        else:
            sub["thermal_wind"] = np.nan

        z500_implied = (R_d / g) * t_aligned * np.log(1000.0 / 500.0)
        z_m = z_height[lat_mask, :]
        z_imp = z500_implied[lat_mask, :]
        valid_hc = np.isfinite(z_m) & np.isfinite(z_imp)
        if valid_hc.sum() >= 100:
            r_hc = np.corrcoef(z_m[valid_hc].ravel(),
                               z_imp[valid_hc].ravel())[0, 1]
            sub["hydrostatic"] = float(max(0.0, min(1.0, r_hc)))
        else:
            sub["hydrostatic"] = np.nan
    else:
        sub["thermal_wind"] = np.nan
        sub["hydrostatic"] = np.nan

    # Composite
    if balance_weights is None:
        balance_weights = config.PCS_BALANCE_WEIGHTS
    total = total_w = 0.0
    for bname, bw in balance_weights.items():
        val = sub.get(bname, np.nan)
        if np.isfinite(val):
            total += bw * val
            total_w += bw
    composite = float(total / total_w) if total_w > 0 else np.nan
    return composite, sub


def autoregressive_stability(global_energy_series: Sequence[float],
                             lead_hours: Sequence[float]) -> float:
    r"""Autoregressive Stability Index (ASI).

    .. math::
        \mathrm{ASI} = \max\!\Big(0,\; 1 - \frac{\gamma\,T}{\ln 2}\Big)

    where :math:`\gamma` is the magnitude of the exponential drift rate of
    global kinetic energy (slope of :math:`\log E` vs lead time) and :math:`T`
    is the forecast-window length.
    """
    E = np.asarray(global_energy_series, dtype=float)
    tau = np.asarray(lead_hours, dtype=float)
    valid = np.isfinite(E) & (E > 0) & np.isfinite(tau)
    E, tau = E[valid], tau[valid]
    if len(E) < 3:
        return np.nan
    try:
        gamma = np.abs(np.polyfit(tau, np.log(E), 1)[0])
        T = tau[-1] - tau[0]
        if T <= 0:
            return np.nan
        return float(max(0.0, 1.0 - gamma * T / np.log(2)))
    except Exception:
        return np.nan


def holistic_score(sfi: float, eff_res: float, tau_d: float,
                   ees: float, pcs: float, asi: float,
                   weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted HMAS composite of the six sub-metrics.

    Missing (``nan``) sub-metrics are dropped and the remaining weights
    renormalised, so a partial set still yields a score.
    """
    if weights is None:
        weights = config.DEFAULT_HMAS_WEIGHTS
    metrics = {"sfi": sfi, "eff_res": eff_res, "tau_d": tau_d,
               "ees": ees, "pcs": pcs, "asi": asi}
    total = total_weight = 0.0
    for key, w in weights.items():
        val = metrics.get(key, np.nan)
        if val is not None and np.isfinite(val):
            total += w * val
            total_weight += w
    return float(total / total_weight) if total_weight > 0 else np.nan


class MetricsComputer:
    """Namespace exposing the metric functions as static methods.

    Retained for backwards API compatibility. The module-level functions are
    the canonical entry points; these simply forward to them.
    """

    spectral_fidelity_index = staticmethod(spectral_fidelity_index)
    effective_resolution = staticmethod(effective_resolution)
    error_doubling_time = staticmethod(error_doubling_time)
    extreme_event_skill = staticmethod(extreme_event_skill)
    physical_consistency_score = staticmethod(physical_consistency_score)
    autoregressive_stability = staticmethod(autoregressive_stability)
    holistic_score = staticmethod(holistic_score)

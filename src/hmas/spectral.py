"""
Isotropic spectral analysis on lat-lon fields.

The power spectrum, the forecast/analysis energy ratio, and the helpers that
derive the Spectral Fidelity Index and effective resolution directly from
pre-computed spectra (avoiding redundant FFTs in batch pipelines).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .stats import align_grids

# Cache of radial-bin index arrays keyed by grid shape, so the integer radius
# array is built only once per resolution.
_RADIAL_BIN_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}


def _get_radial_bins(ny: int, nx: int) -> Tuple[np.ndarray, int]:
    """Return ``(flattened_radius_index, max_radius)`` for a grid shape."""
    key = (ny, nx)
    if key not in _RADIAL_BIN_CACHE:
        cy, cx = ny // 2, nx // 2
        y_idx, x_idx = np.ogrid[-cy:ny - cy, -cx:nx - cx]
        r = np.sqrt(x_idx ** 2 + y_idx ** 2).astype(np.intp)
        max_r = min(cy, cx)
        r_clipped = np.clip(r, 0, max_r - 1)
        _RADIAL_BIN_CACHE[key] = (r_clipped.ravel(), max_r)
    return _RADIAL_BIN_CACHE[key]


def compute_power_spectrum_2d(field_2d: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic (radially averaged) power spectrum of a 2-D lat-lon field.

    A cosine-of-latitude taper is applied before the FFT to suppress the area
    distortion of a regular grid. Radial binning is vectorised with
    ``np.bincount``.

    Returns
    -------
    wavenumbers, power:
        Both exclude the zero (mean) wavenumber.
    """
    ny, nx = field_2d.shape
    lats = np.linspace(90, -90, ny)
    cos_lat = np.clip(np.cos(np.deg2rad(lats)), 0.1, 1.0)
    weighted = field_2d.astype(np.float32) * cos_lat[:, np.newaxis].astype(np.float32)

    F = np.fft.fft2(weighted)
    F_shift = np.fft.fftshift(F)
    power_2d = (np.abs(F_shift) ** 2 / (nx * ny)).ravel()

    r_flat, max_r = _get_radial_bins(ny, nx)

    power_radial = np.bincount(r_flat, weights=power_2d, minlength=max_r)
    counts = np.bincount(r_flat, minlength=max_r).astype(np.float64)
    valid = counts > 0
    power_radial[valid] /= counts[valid]
    wavenumbers = np.arange(max_r)
    return wavenumbers[1:], power_radial[1:]


def compute_spectral_energy_ratio(forecast_field: np.ndarray,
                                  analysis_field: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-wavenumber energy ratio :math:`E_\\mathrm{fcst}(k)/E_\\mathrm{ana}(k)`."""
    forecast_field, analysis_field = align_grids(forecast_field, analysis_field)
    k_f, p_f = compute_power_spectrum_2d(forecast_field)
    k_a, p_a = compute_power_spectrum_2d(analysis_field)
    n = min(len(k_f), len(k_a))
    ratio = p_f[:n] / np.clip(p_a[:n], 1e-30, None)
    return k_f[:n], ratio


def sfi_from_spectra(k_f: np.ndarray, p_f: np.ndarray,
                     k_a: np.ndarray, p_a: np.ndarray,
                     k_min: int = 1, k_max: int = 200) -> float:
    r"""Spectral Fidelity Index from pre-computed spectra.

    .. math::
        \mathrm{SFI} = \max\!\Big(0,\; 1 - \tfrac12\,
        \overline{\big|\log_{10}(E_f/E_a)\big|}\Big),
        \quad k\in[k_{\min}, k_{\max}]

    Returns ``nan`` if fewer than five valid wavenumbers are available.
    """
    n = min(len(k_f), len(k_a))
    k_f, p_f, k_a, p_a = k_f[:n], p_f[:n], k_a[:n], p_a[:n]
    mask = (k_f >= k_min) & (k_f <= k_max) & (p_a > 1e-30) & (p_f > 1e-30)
    if mask.sum() < 5:
        return np.nan
    D = np.mean(np.abs(np.log10(p_f[mask] / p_a[mask])))
    return float(max(0.0, 1.0 - D / 2.0))


def eff_res_from_spectra(k_f: np.ndarray, p_f: np.ndarray,
                         k_a: np.ndarray, p_a: np.ndarray,
                         threshold: float = 0.5,
                         excess_cap: float = 2.0,
                         norm: float = 300.0) -> float:
    r"""Effective resolution from pre-computed spectra.

    The highest wavenumber whose energy ratio lies inside the acceptable band
    ``[threshold, excess_cap]``, normalised by ``norm`` to ``[0, 1]``.

    Both *deficit* (ratio < threshold, i.e. MSE over-smoothing) and *excess*
    (ratio > excess_cap, i.e. energy inflation from instability) are
    penalised, so a model with catastrophic energy drift cannot score 1.0
    merely by inflating small-scale energy.
    """
    n = min(len(k_f), len(k_a))
    ratio = p_f[:n] / np.clip(p_a[:n], 1e-30, None)
    k = k_f[:n]
    within = np.where((ratio >= threshold) & (ratio <= excess_cap)
                      & (k >= 1))[0]
    if len(within) == 0:
        return 0.0
    return min(1.0, float(k[within[-1]]) / norm)

"""Synthetic geostrophically-balanced atmosphere for testing."""

from __future__ import annotations

import numpy as np

from hmas.config import G, OMEGA, R_EARTH

NLAT, NLON = 73, 144
LEAD_HOURS = np.arange(0, 361, 24, dtype=float)  # 0..15 days


def _grids():
    lats = np.linspace(90, -90, NLAT)
    lons = np.linspace(0, 360, NLON, endpoint=False)
    return lats, lons


def truth_fields():
    """Return dict var -> (lead_hours, cube) of a slowly evolving, balanced flow.

    z500 is a planetary-wave geopotential height; u500/v500 are its geostrophic
    winds (so the truth scores high on PCS); t2m/t850 are smooth temperature
    fields. Fields evolve mildly with lead time (phase drift).
    """
    lats, lons = _grids()
    LA = np.deg2rad(lats)[:, None]
    LO = np.deg2rad(lons)[None, :]
    f = 2 * OMEGA * np.sin(LA)
    f_safe = np.where(np.abs(f) > 1e-6, f, np.nan)
    dy = np.deg2rad(180.0 / (NLAT - 1)) * R_EARTH
    dlon = np.deg2rad(360.0 / NLON) * R_EARTH
    dx = np.clip(dlon * np.cos(LA), 1e3, None)

    z_list, u_list, v_list, t2_list, t8_list = [], [], [], [], []
    for h in LEAD_HOURS:
        phase = 0.05 * h / 24.0
        # geopotential height (m): zonal mean + wave-5 pattern
        z = (5500.0 - 400.0 * np.sin(LA)
             + 150.0 * np.cos(5 * LO + phase) * np.cos(LA))
        phi = z * G
        dphi_dy = np.gradient(phi, -dy, axis=0)
        dphi_dx = np.gradient(phi, axis=1) / dx
        u = np.nan_to_num(-dphi_dy / f_safe, nan=0.0)
        v = np.nan_to_num(dphi_dx / f_safe, nan=0.0)
        t2 = 288.0 - 30.0 * np.sin(LA) ** 2 + 2.0 * np.cos(3 * LO + phase)
        t8 = 280.0 - 28.0 * np.sin(LA) ** 2 + 1.5 * np.cos(3 * LO + phase)
        z_list.append(z); u_list.append(u); v_list.append(v)
        t2_list.append(t2); t8_list.append(t8)

    def cube(lst):
        return (LEAD_HOURS.copy(), np.stack(lst).astype(float))

    return {
        "z500": cube(z_list), "u500": cube(u_list), "v500": cube(v_list),
        "t2m": cube(t2_list), "t850": cube(t8_list),
    }, lats


def degrade(cube, smooth, noise, seed):
    """Return a forecast cube: truth blurred in longitude + growing noise."""
    rng = np.random.default_rng(seed)
    lh, c = cube
    out = np.empty_like(c)
    for i in range(c.shape[0]):
        field = c[i]
        # crude longitudinal smoothing (energy loss at small scales)
        if smooth > 0:
            k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
            field = np.apply_along_axis(
                lambda m: np.convolve(np.r_[m[-smooth:], m, m[:smooth]],
                                      k, mode="same")[smooth:-smooth],
                axis=1, arr=field)
        amp = noise * (1.0 + lh[i] / 120.0)
        out[i] = field + rng.normal(0, amp, field.shape)
    return lh.copy(), out


def make_dataset():
    """Two models (one good, one poor) + truth + climatology."""
    truth, lats = truth_fields()
    good = {v: degrade(truth[v], smooth=1, noise=0.5 if v.startswith("t") else 2.0,
                        seed=1) for v in truth}
    poor = {v: degrade(truth[v], smooth=4, noise=2.0 if v.startswith("t") else 8.0,
                       seed=2) for v in truth}
    forecast = {"good_model": {"2023-01-15T00:00:00": good},
                "poor_model": {"2023-01-15T00:00:00": poor}}
    truth_by_init = {"2023-01-15T00:00:00": truth}

    # climatology: time-mean of truth + a plausible std
    clim_mean, clim_std = {}, {}
    for v in ("t2m",):
        c = truth[v][1]
        clim_mean.setdefault("2023-01-15T00:00:00", {})[v] = c.mean(axis=0)
        clim_std.setdefault("2023-01-15T00:00:00", {})[v] = np.clip(
            c.std(axis=0) + 3.0, 0.5, None)
    return forecast, truth_by_init, clim_mean, clim_std, lats

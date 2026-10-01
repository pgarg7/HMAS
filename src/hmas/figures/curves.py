"""Curve exporters for the figures that plot full curves rather than scalar
summaries (Fig 1 power spectra, Fig 2 spectral-energy ratio, Fig 3 RMSE vs lead
time). These figures draw on a precomputed analysis cache
(``cache['raw'][model][init][var]`` with ``lead_hours`` /
``fcst_spectra`` / ``era5_spectra`` / ``rmse``; and ``cache['avg'][model][var]``
with ``lead_hours`` / ``rmse`` / ``rmse_lo`` / ``rmse_hi``).

Run these once on the machine that has the precomputed cache to emit small JSON
artifacts (``spectral_curves.json``, ``spectral_ratio_curves.json``,
``rmse_curves.json``) that can be committed alongside the paper. Reviewers then
regenerate Figs 1-3 from those artifacts with no cache, ERA5, or inference
engine.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_SPECTRAL_VARS = ("u500", "z500")
DEFAULT_SPECTRA_LEADS = OrderedDict([
    ("Day 1", 24), ("Day 5", 120), ("Day 8", 192),
    ("Day 10", 240), ("Day 15", 360),
])
_MATCH_TOL_H = 12.0


def _avg_spectra(raw, model_keys, var, lead_hours, spectrum_key="fcst_spectra"):
    """Average power spectra across inits at one lead."""
    model_spectra = {}
    era5_powers, era5_k = [], None
    for key in model_keys:
        powers, k_ref = [], None
        for _, var_dict in raw.get(key, {}).items():
            entry = var_dict.get(var)
            if entry is None or spectrum_key not in entry:
                continue
            lh = np.asarray(entry["lead_hours"], dtype=float)
            idx = int(np.argmin(np.abs(lh - lead_hours)))
            if np.abs(lh[idx] - lead_hours) > _MATCH_TOL_H:
                continue
            spec = entry[spectrum_key][idx]
            if spec is None:
                continue
            k, p = spec
            if k_ref is None:
                k_ref = np.asarray(k)
            n = min(len(k), len(k_ref))
            powers.append(np.asarray(p)[:n])
            if "era5_spectra" in entry and entry["era5_spectra"][idx] is not None:
                ek, ep = entry["era5_spectra"][idx]
                n_e = min(len(ek), len(k_ref))
                era5_powers.append(np.asarray(ep)[:n_e])
                if era5_k is None:
                    era5_k = np.asarray(ek)[:n_e]
        if powers and k_ref is not None:
            m = min(len(p) for p in powers)
            model_spectra[key] = (k_ref[:m],
                                  np.mean([p[:m] for p in powers], axis=0))
    era5_avg = None
    if era5_powers and era5_k is not None:
        m = min(len(p) for p in era5_powers)
        era5_avg = (era5_k[:m], np.mean([p[:m] for p in era5_powers], axis=0))
    return model_spectra, era5_avg


def _avg_ratio(raw, model_keys, var, lead_hours):
    """Mean (fcst/ERA5) spectral ratio across inits at one lead, per model."""
    out = {}
    for key in model_keys:
        ratios, k_ref = [], None
        for _, var_dict in raw.get(key, {}).items():
            entry = var_dict.get(var)
            if entry is None or "fcst_spectra" not in entry \
                    or "era5_spectra" not in entry:
                continue
            lh = np.asarray(entry["lead_hours"], dtype=float)
            idx = int(np.argmin(np.abs(lh - lead_hours)))
            if np.abs(lh[idx] - lead_hours) > _MATCH_TOL_H:
                continue
            fs, es = entry["fcst_spectra"][idx], entry["era5_spectra"][idx]
            if fs is None or es is None:
                continue
            (k_f, p_f), (_, p_a) = fs, es
            if k_ref is None:
                k_ref = np.asarray(k_f)
            n = min(len(p_f), len(p_a), len(k_ref))
            ratios.append(np.asarray(p_f)[:n]
                          / np.clip(np.asarray(p_a)[:n], 1e-30, None))
        if ratios and k_ref is not None:
            m = min(len(r) for r in ratios)
            out[key] = (k_ref[:m], np.mean([r[:m] for r in ratios], axis=0))
    return out


def _dump(obj, out_path):
    with open(out_path, "w") as f:
        json.dump(obj, f)
    return out_path


def export_spectral_curves(cache, out_path="spectral_curves.json",
                           model_keys: Optional[Sequence[str]] = None,
                           leads=DEFAULT_SPECTRA_LEADS,
                           spectral_vars=DEFAULT_SPECTRAL_VARS):
    """Emit Fig 1 power-spectrum curves: {var: {lead_label: {model|era5:
    {k, power}}}}."""
    raw = cache["raw"]
    model_keys = list(model_keys or sorted(raw.keys()))
    out: Dict = {}
    for var in spectral_vars:
        out[var] = {}
        for label, lh in dict(leads).items():
            ms, era5 = _avg_spectra(raw, model_keys, var, lh)
            panel = {m: {"k": k.tolist(), "power": p.tolist()}
                     for m, (k, p) in ms.items()}
            if era5 is not None:
                panel["era5"] = {"k": era5[0].tolist(),
                                 "power": era5[1].tolist()}
            out[var][label] = panel
    return _dump(out, out_path)


def export_spectral_ratio_curves(cache,
                                 out_path="spectral_ratio_curves.json",
                                 model_keys: Optional[Sequence[str]] = None,
                                 days=(1, 5, 15),
                                 spectral_vars=DEFAULT_SPECTRAL_VARS):
    """Emit Fig 2 ratio curves: {var: {day: {model: {k, ratio}}}}."""
    raw = cache["raw"]
    model_keys = list(model_keys or sorted(raw.keys()))
    out: Dict = {}
    for var in spectral_vars:
        out[var] = {}
        for d in days:
            r = _avg_ratio(raw, model_keys, var, d * 24)
            out[var][str(d)] = {m: {"k": k.tolist(), "ratio": v.tolist()}
                                for m, (k, v) in r.items()}
    return _dump(out, out_path)


def export_rmse_curves(cache, out_path="rmse_curves.json",
                       model_keys: Optional[Sequence[str]] = None,
                       var_list: Optional[Sequence[str]] = None):
    """Emit Fig 3 RMSE-vs-lead curves with CIs: {var: {model: {lead_hours,
    rmse, rmse_lo, rmse_hi}}}."""
    avg = cache["avg"]
    model_keys = list(model_keys or sorted(avg.keys()))
    if var_list is None:
        var_list = sorted({v for k in model_keys for v in avg.get(k, {})})
    out: Dict = {}
    for var in var_list:
        out[var] = {}
        for key in model_keys:
            a = avg.get(key, {}).get(var)
            if a is None:
                continue
            rec = {"lead_hours": np.asarray(a["lead_hours"]).tolist(),
                   "rmse": np.asarray(a["rmse"]).tolist()}
            for ci in ("rmse_lo", "rmse_hi"):
                if a.get(ci) is not None:
                    rec[ci] = np.asarray(a[ci]).tolist()
            out[var][key] = rec
    return _dump(out, out_path)

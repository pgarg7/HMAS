"""Figures 1-2 -- spectral diagnostics.

Both read curve artifacts produced by :mod:`hmas.figures.curves`:
  * Fig 1 ``spectral_comparison`` <- ``spectral_curves.json``
  * Fig 2 ``spectral_ratio``      <- ``spectral_ratio_curves.json``

These plot power spectra / spectral-energy ratios that live in the precomputed
cache, not in the scalar metric JSONs, hence the separate curve artifacts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import style
from ._io import DataLike, load

_VAR_LABELS = {
    "u500": r"U500 -- Kinetic energy ($E(\ell)\sim\ell^{-3}/\ell^{-5/3}$)",
    "z500": r"Z500 -- Geopotential height ($E(\ell)\sim\ell^{-4}$)",
}


def spectral_comparison(curves: DataLike, title: Optional[str] = None):
    """Fig 1: power spectra of each model vs ERA5, per variable (rows) and
    lead time (columns). ``curves`` is ``spectral_curves.json`` or its dict."""
    import matplotlib.pyplot as plt

    data = load(curves)
    var_list = [v for v in ("u500", "z500") if v in data] or list(data)
    lead_labels = list(data[var_list[0]].keys())
    ncol = len(lead_labels)

    fig, axes = plt.subplots(len(var_list), ncol,
                             figsize=(5 * ncol, 4.5 * len(var_list)),
                             squeeze=False)
    for row, var in enumerate(var_list):
        for col, lab in enumerate(lead_labels):
            ax = axes[row][col]
            panel = data[var].get(lab, {})
            for key in style.order_models(
                    [m for m in panel if m != "era5"]):
                k = np.asarray(panel[key]["k"], dtype=float)
                p = np.asarray(panel[key]["power"], dtype=float)
                ax.loglog(k, p, style.LINESTYLES.get(key, "-"), lw=1.4,
                          color=style.model_color(key),
                          label=style.model_label(key), alpha=0.85)
            if "era5" in panel:
                ax.loglog(np.asarray(panel["era5"]["k"], dtype=float),
                          np.asarray(panel["era5"]["power"], dtype=float),
                          "k-", lw=2.5, label="ERA5", zorder=10)
            ax.set_xlabel(r"Wavenumber $k$")
            ax.set_ylabel("Power spectral density")
            ax.set_title(f"{_VAR_LABELS.get(var, var.upper())}\n{lab}")
            ax.set_xlim(1, 360)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=10,
               frameon=True, edgecolor="gray")
    if title is None:
        title = ("Power spectra: AI weather models vs ERA5\n"
                 "Spectral energy loss at high wavenumbers confirms "
                 "loss-induced bias")
    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    return fig


def spectral_ratio(curves: DataLike, var: str = "u500",
                   title: Optional[str] = None):
    """Fig 2: mean forecast/ERA5 spectral-energy ratio vs wavenumber, one panel
    per lead day. ``curves`` is ``spectral_ratio_curves.json`` or its dict; pass
    the ``var`` to plot (the paper uses u500 and z500)."""
    import matplotlib.pyplot as plt

    data = load(curves)
    if var not in data:
        raise KeyError(f"{var!r} not in ratio curves; have {list(data)}")
    days = list(data[var].keys())
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 4.5),
                             squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, days):
        panel = data[var][d]
        for key in style.order_models(list(panel)):
            k = np.asarray(panel[key]["k"], dtype=float)
            r = np.asarray(panel[key]["ratio"], dtype=float)
            ax.semilogx(k, r, style.LINESTYLES.get(key, "-"), lw=1.4,
                        color=style.model_color(key),
                        label=style.model_label(key), alpha=0.85)
        ax.axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.6)
        ax.set_xlabel(r"Wavenumber $k$")
        ax.set_ylabel("Spectral energy ratio (model / ERA5)")
        ax.set_title(f"Day {d}")
        ax.set_xlim(1, 360)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=6, fontsize=10,
               frameon=True, edgecolor="gray")
    if title is None:
        title = (f"Spectral energy ratio for {var.upper()}\n"
                 r"ratio < 1 at high $k$ confirms MSE spectral bias")
    fig.suptitle(title, fontsize=15, y=1.03)
    fig.tight_layout()
    return fig

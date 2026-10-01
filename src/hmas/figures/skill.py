"""Figures 3-4 -- deterministic skill vs lead time.

  * Fig 3 ``rmse_vs_leadtime`` <- ``rmse_curves.json`` (from hmas.figures.curves)
  * Fig 4 ``acc_vs_leadtime``  <- ``acc_metrics.json`` (committed scalar artifact)

Fig 4 reproduces directly from the committed per-lead ACC snapshots
(ACC_{var}_day{N}); Fig 3 needs the RMSE-vs-lead curves from the precomputed
cache, exported via :func:`hmas.figures.curves.export_rmse_curves`.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

from . import style
from ._io import DataLike, load

_UNITS = {"z500": "m", "t850": "K", "t2m": "K", "u500": "m/s",
          "v500": "m/s", "q700": "kg/kg"}


def rmse_vs_leadtime(curves: DataLike,
                     var_list: Optional[Sequence[str]] = None,
                     title: Optional[str] = None):
    """Fig 3: global RMSE vs lead time, one panel per variable, with CI bands
    when present. ``curves`` is ``rmse_curves.json`` or its dict."""
    import matplotlib.pyplot as plt

    data = load(curves)
    var_list = list(var_list or data.keys())
    fig, axes = plt.subplots(1, len(var_list),
                             figsize=(4.5 * len(var_list), 5), squeeze=False)
    axes = axes[0]
    for ax, var in zip(axes, var_list):
        panel = data.get(var, {})
        for key in style.order_models(list(panel)):
            rec = panel[key]
            h = np.asarray(rec["lead_hours"], dtype=float) / 24.0
            rmse = np.asarray(rec["rmse"], dtype=float)
            ax.plot(h, rmse, style.LINESTYLES.get(key, "-"), lw=1.6,
                    color=style.model_color(key),
                    label=style.model_label(key))
            lo, hi = rec.get("rmse_lo"), rec.get("rmse_hi")
            if lo is not None and hi is not None:
                lo, hi = np.asarray(lo, float), np.asarray(hi, float)
                m = np.isfinite(lo) & np.isfinite(hi)
                ax.fill_between(h, lo, hi, where=m, alpha=0.12,
                                color=style.model_color(key), lw=0)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(f"RMSE ({_UNITS.get(var, '')})")
        ax.set_title(var.upper())
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=10,
               frameon=True, edgecolor="gray")
    fig.suptitle(title or "Global RMSE vs lead time", fontsize=15, y=1.02)
    fig.tight_layout()
    return fig


def _parse_acc(metrics):
    """{model: {ACC_{var}_day{N}: val}} -> {var: {model: (days, accs)}}."""
    pat = re.compile(r"ACC_([A-Za-z0-9]+)_day(\d+)")
    by_var = defaultdict(lambda: defaultdict(dict))
    for model, d in metrics.items():
        for k, v in d.items():
            m = pat.match(k)
            if m:
                by_var[m.group(1)][model][int(m.group(2))] = v
    out = {}
    for var, models in by_var.items():
        out[var] = {}
        for model, dd in models.items():
            days = sorted(dd)
            out[var][model] = (np.array(days, float),
                               np.array([dd[x] for x in days], float))
    return out


def acc_vs_leadtime(acc_metrics: DataLike,
                    var_list: Optional[Sequence[str]] = None,
                    title: Optional[str] = None):
    """Fig 4: anomaly correlation coefficient vs lead time, one panel per
    variable, with the ACC=0.6 useful-skill line. ``acc_metrics`` is
    ``acc_metrics.json`` or its dict."""
    import matplotlib.pyplot as plt

    parsed = _parse_acc(load(acc_metrics))
    var_list = list(var_list or ["z500", "t850", "t2m"])
    var_list = [v for v in var_list if v in parsed] or list(parsed)
    fig, axes = plt.subplots(1, len(var_list),
                             figsize=(4.5 * len(var_list), 5), squeeze=False)
    axes = axes[0]
    for ax, var in zip(axes, var_list):
        for key in style.order_models(list(parsed[var])):
            days, accs = parsed[var][key]
            ax.plot(days, accs, style.LINESTYLES.get(key, "-"),
                    marker="o", ms=4, lw=1.6, color=style.model_color(key),
                    label=style.model_label(key))
        ax.axhline(0.6, color="k", ls=":", lw=1.2, alpha=0.6)
        ax.text(0.5, 0.585, "Useful skill (ACC=0.6)", fontsize=9, alpha=0.7)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("ACC")
        ax.set_title(f"{var.upper()} Anomaly Correlation")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=10,
               frameon=True, edgecolor="gray")
    fig.suptitle(title or "Anomaly Correlation Coefficient vs lead time",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    return fig

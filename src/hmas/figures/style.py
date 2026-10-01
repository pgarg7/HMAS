"""Shared style for the paper figures: per-model colours, line styles, display
labels, the HMAS sub-metric ordering/labels, and a savefig helper, kept fixed
so regenerated figures match the published versions.
"""
from __future__ import annotations

from typing import Dict, List

# Per-model colours (consistent across every figure).
COLORS: Dict[str, str] = {
    "atlas":     "#e41a1c",
    "aifs":      "#377eb8",
    "aifsens":   "#4daf4a",
    "aurora":    "#984ea3",
    "fcn3":      "#ff7f00",
    "fengwu":    "#a65628",
    "fuxi":      "#f781bf",
    "graphcast": "#999999",
    "pangu":     "#e6ab02",
    "sfno":      "#66c2a5",
    "era5":      "#000000",
}

# Per-model line styles, grouped loosely by architecture family.
LINESTYLES: Dict[str, str] = {
    "aifs": "-", "aifsens": "--", "graphcast": "-",
    "fcn3": "--", "sfno": "--",
    "aurora": "-.", "fengwu": "-.",
    "atlas": "-", "fuxi": ":", "pangu": ":",
}

# Display labels including architecture tag.
MODEL_LABELS: Dict[str, str] = {
    "atlas": "Atlas (DiT)", "aifs": "AIFS (GNN)",
    "aifsens": "AIFS-ENS (GNN)", "aurora": "Aurora (ViT)",
    "fcn3": "FCN3 (SFNO)", "fengwu": "FengWu (ViT)",
    "fuxi": "FuXi (U-Trans)", "graphcast": "GraphCast (GNN)",
    "pangu": "Pangu (3D-ET)", "sfno": "SFNO (SHT)",
}

# HMAS sub-metric ordering, internal keys, and display (LaTeX) labels.
METRIC_KEYS: List[str] = ["sfi", "eff_res", "tau_d", "ees", "pcs", "asi"]
METRIC_LABELS: List[str] = ["SFI", r"$\ell_{eff}$", r"$\tau_d$",
                            "EES", "PCS", "ASI"]

# A reasonable default ordering for models when a figure iterates them.
MODEL_ORDER: List[str] = [
    "aifs", "aifsens", "graphcast", "pangu", "fuxi",
    "aurora", "fengwu", "fcn3", "sfno", "atlas",
]


def model_label(key: str) -> str:
    return MODEL_LABELS.get(key, key)


def model_color(key: str) -> str:
    return COLORS.get(key, "#444444")


def order_models(keys) -> List[str]:
    """Return ``keys`` in the canonical model order, with unknowns appended."""
    keys = list(keys)
    known = [m for m in MODEL_ORDER if m in keys]
    extra = [m for m in keys if m not in MODEL_ORDER]
    return known + extra


def savefig(fig, path_stem: str, dpi: int = 300, also_png: bool = True) -> None:
    """Save ``fig`` as a vector PDF (rasterizing pcolormesh/images to keep the
    file small) and, optionally, a PNG preview. ``path_stem`` carries no
    extension, e.g. ``".../fig11_holistic_radar_day5"``.
    """
    for ax in fig.get_axes():
        for coll in ax.collections:
            coll.set_rasterized(True)
        for img in ax.images:
            img.set_rasterized(True)
    fig.savefig(f"{path_stem}.pdf", dpi=dpi, bbox_inches="tight")
    if also_png:
        fig.savefig(f"{path_stem}.png", dpi=min(dpi, 150), bbox_inches="tight")

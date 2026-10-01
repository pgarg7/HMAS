"""
Data-source abstractions.

HMAS never assumes a file format or inference engine. Instead it reads through
two abstract interfaces:

* :class:`ForecastStore` -- yields forecast fields for a model / init time /
  variable / lead time.
* :class:`VerificationSource` -- yields the verifying truth (e.g. a reanalysis)
  and, optionally, a climatology (mean + std) for ACC and EES.

Bring your own data by subclassing these, or use the in-memory
:class:`ArrayForecastStore` / :class:`ArrayVerification` for arrays you already
hold. An optional :class:`NetCDFForecastStore` is provided for the common
on-disk layout (requires ``xarray``).
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
Series = Tuple[Optional[np.ndarray], Optional[np.ndarray]]  # (lead_hours, cube)


# ---------------------------------------------------------------------------
# Forecast store
# ---------------------------------------------------------------------------
class ForecastStore(abc.ABC):
    """Abstract forecast reader.

    Subclasses must implement :meth:`list_models`, :meth:`init_times` and
    :meth:`get_series`. :meth:`get_field` and :meth:`latitudes` have working
    defaults derived from those.
    """

    @abc.abstractmethod
    def list_models(self) -> List[str]:
        """Return the available model identifiers."""

    @abc.abstractmethod
    def init_times(self, model: str) -> List[str]:
        """Return ISO init-time strings available for ``model``."""

    @abc.abstractmethod
    def get_series(self, model: str, init_time: str, variable: str) -> Series:
        """Return ``(lead_hours, cube)`` for one (model, init, variable).

        ``cube`` has shape ``(n_lead, n_lat, n_lon)`` ordered North->South in
        latitude. Return ``(None, None)`` if unavailable.
        """

    def get_field(self, model: str, init_time: str, variable: str,
                  lead_hours: float, tolerance: float = 12.0
                  ) -> Optional[Array]:
        """Return the 2-D field nearest ``lead_hours`` (within ``tolerance``)."""
        lh, cube = self.get_series(model, init_time, variable)
        if lh is None or cube is None:
            return None
        idx = int(np.argmin(np.abs(lh - lead_hours)))
        if np.abs(lh[idx] - lead_hours) > tolerance:
            return None
        return np.asarray(cube[idx], dtype=float)

    def latitudes(self, model: str, init_time: str) -> Optional[Array]:
        """Return the latitude vector, or ``None`` to assume ``[90..-90]``."""
        return None

    # Backwards-compatible alias for the original method name.
    def get_all_steps(self, model: str, init_time: str, variable: str) -> Series:
        return self.get_series(model, init_time, variable)


class ArrayForecastStore(ForecastStore):
    """In-memory store backed by nested dicts of numpy arrays.

    Parameters
    ----------
    data:
        ``{model: {init_time: {variable: (lead_hours, cube)}}}`` where
        ``lead_hours`` is 1-D ``(n_lead,)`` and ``cube`` is
        ``(n_lead, n_lat, n_lon)``.
    latitudes:
        Optional global latitude vector (degrees, North->South).
    """

    def __init__(self,
                 data: Dict[str, Dict[str, Dict[str, Tuple[Array, Array]]]],
                 latitudes: Optional[Sequence[float]] = None):
        self._data = data
        self._lats = (None if latitudes is None
                      else np.asarray(latitudes, dtype=float))

    def list_models(self) -> List[str]:
        return sorted(self._data.keys())

    def init_times(self, model: str) -> List[str]:
        return sorted(self._data.get(model, {}).keys())

    def get_series(self, model: str, init_time: str, variable: str) -> Series:
        entry = self._data.get(model, {}).get(init_time, {}).get(variable)
        if entry is None:
            return None, None
        lh, cube = entry
        return np.asarray(lh, dtype=float), np.asarray(cube, dtype=float)

    def latitudes(self, model: str, init_time: str) -> Optional[Array]:
        return self._lats


# ---------------------------------------------------------------------------
# Verification source
# ---------------------------------------------------------------------------
class VerificationSource(abc.ABC):
    """Abstract truth + climatology reader."""

    @abc.abstractmethod
    def get_truth(self, variable: str, init_time: str) -> Series:
        """Return ``(lead_hours, cube)`` of verifying truth for one init."""

    def get_climatology_mean(self, variable: str,
                             init_time: str) -> Optional[Array]:
        """Climatological mean field for ACC / EES, or ``None``."""
        return None

    def get_climatology_std(self, variable: str,
                            init_time: str) -> Optional[Array]:
        """Climatological standard-deviation field for EES, or ``None``."""
        return None

    def latitudes(self) -> Optional[Array]:
        return None


class ArrayVerification(VerificationSource):
    """In-memory verification backed by nested dicts of numpy arrays.

    Parameters
    ----------
    truth:
        ``{init_time: {variable: (lead_hours, cube)}}``.
    climatology_mean, climatology_std:
        Optional ``{init_time: {variable: field_2d}}`` used by ACC and EES.
    latitudes:
        Optional global latitude vector.
    """

    def __init__(self,
                 truth: Dict[str, Dict[str, Tuple[Array, Array]]],
                 climatology_mean: Optional[Dict[str, Dict[str, Array]]] = None,
                 climatology_std: Optional[Dict[str, Dict[str, Array]]] = None,
                 latitudes: Optional[Sequence[float]] = None):
        self._truth = truth
        self._clim_mean = climatology_mean or {}
        self._clim_std = climatology_std or {}
        self._lats = (None if latitudes is None
                      else np.asarray(latitudes, dtype=float))

    def get_truth(self, variable: str, init_time: str) -> Series:
        entry = self._truth.get(init_time, {}).get(variable)
        if entry is None:
            return None, None
        lh, cube = entry
        return np.asarray(lh, dtype=float), np.asarray(cube, dtype=float)

    def get_climatology_mean(self, variable, init_time):
        field = self._clim_mean.get(init_time, {}).get(variable)
        return None if field is None else np.asarray(field, dtype=float)

    def get_climatology_std(self, variable, init_time):
        field = self._clim_std.get(init_time, {}).get(variable)
        return None if field is None else np.asarray(field, dtype=float)

    def latitudes(self) -> Optional[Array]:
        return self._lats


# ---------------------------------------------------------------------------
# Optional NetCDF reader (requires xarray)
# ---------------------------------------------------------------------------
class NetCDFForecastStore(ForecastStore):
    """Forecast store reading per-(model, init) NetCDF files via ``xarray``.

    Expected layout::

        {root}/{model}/{init_tag}*.nc

    where ``init_tag`` is ``YYYYMMDDTHHMMSS`` and each file holds the
    variables as ``(lead_time, lat, lon)`` data arrays with a ``lead_time``
    coordinate in hours. This mirrors the common on-disk convention but makes
    no reference to any specific inference engine.

    Parameters
    ----------
    root:
        Directory containing one subdirectory per model.
    geopotential_vars:
        Variable names to auto-convert from geopotential (m^2 s^-2) to
        geopotential height (m) when the field magnitude exceeds 10000.
        Defaults to ``{"z500"}``.
    """

    def __init__(self, root: str, geopotential_vars: Optional[set] = None):
        import os
        import glob
        self._os, self._glob = os, glob
        self.root = root
        self._geo_vars = ({"z500"} if geopotential_vars is None
                          else set(geopotential_vars))
        self._cache: Dict[tuple, object] = {}
        self._scan()

    def _scan(self) -> None:
        os, glob = self._os, self._glob
        self._files: Dict[str, List[str]] = {}
        if not os.path.isdir(self.root):
            return
        for model in sorted(os.listdir(self.root)):
            mdir = os.path.join(self.root, model)
            if os.path.isdir(mdir):
                ncs = sorted(glob.glob(os.path.join(mdir, "*.nc")))
                if ncs:
                    self._files[model] = ncs

    @staticmethod
    def _init_to_tag(init_time: str) -> str:
        from datetime import datetime
        return datetime.fromisoformat(init_time).strftime("%Y%m%dT%H%M%S")

    def _open(self, model: str, init_time: str):
        import xarray as xr  # noqa: lazy import
        os, glob = self._os, self._glob
        key = (model, init_time)
        if key in self._cache:
            return self._cache[key]
        # Evict other inits' datasets so a long run holds only the current
        # forecast file open (each is multi-GB), not all 30 at once.
        for k in list(self._cache):
            if k != key:
                ds_old = self._cache.pop(k)
                try:
                    if ds_old is not None:
                        ds_old.close()
                except Exception:
                    pass
        tag = self._init_to_tag(init_time)
        matches = sorted(glob.glob(
            os.path.join(self.root, model, f"{tag}*.nc")))
        ds = xr.open_dataset(matches[-1]) if matches else None
        self._cache[key] = ds
        return ds

    def list_models(self) -> List[str]:
        return sorted(self._files.keys())

    def init_times(self, model: str) -> List[str]:
        from datetime import datetime
        os = self._os
        out = []
        for fpath in self._files.get(model, []):
            tag = os.path.basename(fpath).split("_")[0].split(".")[0]
            try:
                out.append(datetime.strptime(
                    tag[:15], "%Y%m%dT%H%M%S").isoformat())
            except ValueError:
                pass
        return sorted(set(out))

    def get_series(self, model: str, init_time: str, variable: str) -> Series:
        ds = self._open(model, init_time)
        if ds is None or variable not in ds:
            return None, None
        da = ds[variable]
        if "lead_time" not in da.dims:
            return None, None
        lh = da["lead_time"].values.astype(float)
        cube = da.values
        while cube.ndim > 3:
            cube = cube[:, 0]
        cube = cube.astype(float)
        if variable in self._geo_vars and np.nanmean(np.abs(cube)) > 10000:
            from .config import G
            cube = cube / G
        return lh, cube

    def latitudes(self, model: str, init_time: str) -> Optional[Array]:
        ds = self._open(model, init_time)
        if ds is None:
            return None
        for name in ("lat", "latitude"):
            if name in ds.coords:
                return ds[name].values.astype(float)
        return None

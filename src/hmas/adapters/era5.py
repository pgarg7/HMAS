"""hmas.adapters.era5 -- VerificationSource adapters for an on-disk ERA5
layout: per-init verification files plus a day-of-year t2m climatology,
and a ChaosBench spatial-climatology zarr. Reads xarray/netCDF/zarr directly,
with no dependency on any inference engine.

Layout assumed:
  verif_dir : per-init files  era5_verif_YYYYMMDD.nc  with a `valid_time`
              (or `time`) dimension whose values are the forecast valid times
              (init + lead). Variables either use short names directly
              (z500, t850, u500, v500, t2m, q700) or raw CDS names + a level
              dimension.
  clim_dir  : a single file carrying a `dayofyear` (and usually `hour`) dim;
              holds the t2m climatology used for Extreme Event Skill.

Usage:

    from hmas import HMASEvaluator, NetCDFForecastStore, VariableRoles
    from hmas.adapters.era5 import ERA5VerificationSource

    store = NetCDFForecastStore(".../forecasts")
    verif = ERA5VerificationSource(verif_dir=".../era5_verif",
                                   clim_dir=".../era5_data")
    res = HMASEvaluator(store, verif).evaluate(eval_days=[3, 5, 15])
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..stores import VerificationSource

# short name -> CDS fallbacks, pressure level (hPa), z500 -> height scale.
_VARNAME_MAP = {
    "u500": {"cds": ["u", "U", "u_component_of_wind"], "level": 500},
    "v500": {"cds": ["v", "V", "v_component_of_wind"], "level": 500},
    "z500": {"cds": ["z", "Z", "geopotential"], "level": 500,
             "scale": 1.0 / 9.80665},
    "t2m":  {"cds": ["t2m", "2t", "VAR_2T", "2m_temperature"], "level": None},
    "t850": {"cds": ["t", "T", "temperature"], "level": 850},
    "q700": {"cds": ["q", "Q", "specific_humidity"], "level": 700},
}
_TIME_DIMS = ("valid_time", "time")
_LEVEL_DIMS = ("level", "pressure_level", "pressure", "isobaricInhPa",
               "plev", "lev")
_T2M_CLIM_NAMES = ("2m_temperature", "t2m", "2t")


class ERA5VerificationSource(VerificationSource):
    """Reads ERA5 truth + t2m climatology for the hmas evaluator.

    Parameters
    ----------
    verif_dir, clim_dir:
        Directories described in the module docstring. ``clim_dir`` may be
        ``None`` (then EES is skipped and HMAS renormalizes around it).
    lead_hours:
        Lead grid (hours) to request from each verification file. Defaults to
        6..384 every 6 h, a superset of the 6..366 forecast grid; only leads
        with a matching valid time are returned, so an over-wide grid is safe.
    clim_window_days:
        Half-window (days) for the day-of-year climatology average.
    extreme_var:
        Which role variable carries climatology (only this one returns
        non-None from get_climatology_*). Defaults to "t2m".
    tolerance_hours:
        Nearest-time match tolerance (hours).
    """

    def __init__(self,
                 verif_dir: str,
                 clim_dir: Optional[str] = None,
                 lead_hours: Optional[np.ndarray] = None,
                 clim_window_days: int = 15,
                 extreme_var: str = "t2m",
                 tolerance_hours: int = 6,
                 climatology=None):
        self.verif_dir = verif_dir
        self.clim_dir = clim_dir
        # Optional external climatology provider exposing
        #   fields(role: str, init_time: str) -> (mean_2d, std_2d) on the
        #   forecast grid. If given, it takes precedence over the built-in
        #   dayofyear-NetCDF reader (e.g. ChaosBenchClimatology below).
        self.climatology = climatology
        self.lead_hours = (np.arange(6, 385, 6, dtype=float)
                           if lead_hours is None
                           else np.asarray(lead_hours, dtype=float))
        self.clim_window_days = clim_window_days
        self.extreme_var = extreme_var
        self.tol = np.timedelta64(tolerance_hours, "h")

        # filename index: 'YYYYMMDD' -> path ; plus a list for fallback scans
        self._by_date: Dict[str, str] = {}
        self._all_files: List[str] = []
        for f in sorted(glob.glob(os.path.join(verif_dir, "*.nc"))):
            self._all_files.append(f)
            m = re.search(r"(\d{8})", os.path.basename(f))
            if m:
                self._by_date.setdefault(m.group(1), f)

        self._ds_cache: Dict[str, object] = {}        # path -> open Dataset
        self._truth_cache: Dict[Tuple[str, str], Tuple] = {}
        self._open_init = None      # init whose datasets/cubes are resident
        self._clim_path = None      # protected from eviction (shared clim file)
        self._clim_ds = None
        self._clim_loaded = False

    # ------------------------------------------------------------------ utils
    def _open(self, path: str):
        """Open a verification file, returning the Dataset or ``None`` if it
        cannot be read. A single unreadable file must not abort a whole run, so
        failures are logged and skipped rather than raised. Tries the default
        engine, then h5netcdf as a fallback for HDF5 locking/backend issues."""
        import os as _os
        # Best-effort: disable HDF5 file locking (common on Lustre/GPFS/NFS).
        _os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        if path in self._ds_cache:
            return self._ds_cache[path]
        import xarray as xr
        last_err = None
        for engine in (None, "h5netcdf"):
            try:
                kw = {} if engine is None else {"engine": engine}
                self._ds_cache[path] = xr.open_dataset(path, chunks=None, **kw)
                return self._ds_cache[path]
            except Exception as e:               # noqa: skip bad file, keep run
                last_err = e
        self._warn(f"could not open {_os.path.basename(path)} "
                   f"({type(last_err).__name__}: {last_err}); skipping")
        self._ds_cache[path] = None
        return None

    def _warn(self, msg: str) -> None:
        import sys
        print(f"[hmas.era5] WARNING: {msg}", flush=True, file=sys.stderr)

    @staticmethod
    def _time_dim(ds) -> Optional[str]:
        for d in _TIME_DIMS:
            if d in ds.dims or d in ds.coords:
                return d
        return None

    @staticmethod
    def _level_dim(ds) -> Optional[str]:
        for d in _LEVEL_DIMS:
            if d in ds.dims or d in ds.coords:
                return d
        return None

    def _resolve_var(self, ds, variable: str) -> Optional[str]:
        if variable in ds.data_vars:           # per-date files use short names
            return variable
        for name in _VARNAME_MAP.get(variable, {}).get("cds", []):
            if name in ds.data_vars:
                return name
        return None

    def _files_for_init(self, init_time: str) -> List[str]:
        tag = datetime.fromisoformat(init_time).strftime("%Y%m%d")
        if tag in self._by_date:
            return [self._by_date[tag]]
        return list(self._all_files)            # fallback: scan everything

    def _evict_other_inits(self, init_time: str) -> None:
        """Keep only the current init's datasets and truth cubes resident
        (plus the shared climatology), so memory stays bounded across a long
        multi-init run instead of accumulating ~500 MB per (variable, init)."""
        if init_time == self._open_init:
            return
        for path in list(self._ds_cache):
            if path == self._clim_path:
                continue
            ds = self._ds_cache.pop(path)
            try:
                if ds is not None:
                    ds.close()
            except Exception:
                pass
        self._truth_cache.clear()
        self._open_init = init_time

    # ----------------------------------------------------------------- truth
    def get_truth(self, variable: str, init_time: str):
        self._evict_other_inits(init_time)
        key = (variable, init_time)
        if key in self._truth_cache:
            return self._truth_cache[key]

        init_dt = datetime.fromisoformat(init_time)
        targets = np.array([np.datetime64(init_dt + timedelta(hours=int(h)))
                            for h in self.lead_hours])
        spec = _VARNAME_MAP.get(variable, {})
        level = spec.get("level")
        scale = spec.get("scale", 1.0)

        for path in self._files_for_init(init_time):
            ds = self._open(path)
            if ds is None:
                continue
            vname = self._resolve_var(ds, variable)
            tdim = self._time_dim(ds)
            if vname is None or tdim is None:
                continue
            tvals = ds[tdim].values
            in_range = ((targets >= tvals.min() - self.tol) &
                        (targets <= tvals.max() + self.tol))
            if not in_range.any():
                continue
            try:
                da = ds[vname].sel(**{tdim: targets[in_range]},
                                   method="nearest", tolerance=self.tol)
            except (KeyError, ValueError):
                continue

            if level is not None:
                ldim = self._level_dim(ds)
                if ldim is not None and ldim in da.dims:
                    lv = da[ldim].values
                    sel = (level * 100 if np.nanmax(lv) > 2000 else level)
                    tolv = 5000 if np.nanmax(lv) > 2000 else 50
                    try:
                        da = da.sel(**{ldim: sel}, method="nearest",
                                    tolerance=tolv)
                    except (KeyError, ValueError):
                        continue

            data = np.asarray(da.values).squeeze()
            if data.ndim == 2:
                data = data[None]
            if data.ndim != 3:
                continue
            # z500 geopotential -> height (skip if already in metres)
            if scale != 1.0 and not (variable == "z500"
                                     and np.nanmean(np.abs(data)) < 10000):
                data = data * scale

            lh = self.lead_hours[in_range]
            cube = data.astype(float)
            out = (np.asarray(lh, dtype=float), cube)
            self._truth_cache[key] = out
            return out

        self._truth_cache[key] = (None, None)
        return None, None

    # ----------------------------------------------------------- climatology
    def _ensure_clim(self):
        if self._clim_loaded:
            return
        self._clim_loaded = True
        search = []
        if self.clim_dir:
            search += sorted(glob.glob(os.path.join(self.clim_dir, "*.nc")))
        search += self._all_files            # clim may live among verif files
        for f in search:
            try:
                ds = self._open(f)
            except Exception:
                continue
            if ds is not None and "dayofyear" in ds.dims:
                self._clim_ds = ds
                self._clim_path = f      # protect from per-init eviction
                return

    def _clim_fields(self, init_time: str):
        self._ensure_clim()
        ds = self._clim_ds
        if ds is None:
            return None, None
        dt = datetime.fromisoformat(init_time)
        doy = dt.timetuple().tm_yday
        hour = (dt.hour // 6) * 6
        if "hour" in ds.dims:
            try:
                dsh = ds.sel(hour=hour)
            except (KeyError, ValueError):
                dsh = ds.sel(hour=0)
        else:
            dsh = ds

        max_doy = int(ds.dayofyear.max().values)
        avail = set(int(x) for x in ds.dayofyear.values)
        win = sorted({((doy + d - 1) % max_doy) + 1
                      for d in range(-self.clim_window_days,
                                     self.clim_window_days + 1)} & avail)
        if not win:
            return None, None
        vname = next((n for n in _T2M_CLIM_NAMES if n in dsh.data_vars), None)
        if vname is None:
            return None, None
        sub = dsh[vname].sel(dayofyear=win)
        mean = sub.mean(dim="dayofyear").values.astype(float)
        std = np.clip(sub.std(dim="dayofyear").values.astype(float), 0.5, None)
        return mean, std

    def get_climatology_mean(self, variable: str, init_time: str):
        if self.climatology is not None:
            return self.climatology.fields(variable, init_time)[0]
        if variable != self.extreme_var:
            return None
        return self._clim_fields(init_time)[0]

    def get_climatology_std(self, variable: str, init_time: str):
        if self.climatology is not None:
            return self.climatology.fields(variable, init_time)[1]
        if variable != self.extreme_var:
            return None
        return self._clim_fields(init_time)[1]

    # ----------------------------------------------------------------- grid
    def latitudes(self):
        for path in self._all_files:
            ds = self._open(path)
            if ds is None:
                continue
            for name in ("latitude", "lat"):
                if name in ds.coords:
                    return ds[name].values.astype(float)
        return None

    def close(self):
        for ds in self._ds_cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self._ds_cache.clear()


class ChaosBenchClimatology:
    """Climatology provider for a ChaosBench spatial-climatology zarr store.

    The store has dims (doy, param, lat, lon) and data variables ``mean`` and
    ``sigma``, where ``param`` is a string code like ``"t-850"`` / ``"z-500"``
    (``{variable}-{level}``). It is typically on a coarse grid (e.g. 1.5 deg,
    121 x 240), so ``mean``/``sigma`` are bilinearly regridded onto the
    forecast grid before being returned, because the hmas evaluator aligns
    fields by truncation, not interpolation.

    Parameters
    ----------
    zarr_path:
        Path to the ``.zarr`` store.
    target_lat, target_lon:
        1-D coordinate vectors of the forecast grid to interpolate onto. Get
        them from the store/verification (lat North->South, lon 0..360).
    role_to_param:
        Maps an hmas role variable to a ``param`` code in the store, e.g.
        ``{"t2m": "t-1000", "t850": "t-850"}``. A role absent from this map
        returns ``(None, None)`` (that sub-metric is then skipped).
    sigma_floor:
        Lower clip on the standard deviation (default 0.5, matching the 0.5 K
        floor of the built-in t2m climatology reader); set per the units of
        the chosen variable.
    """

    def __init__(self, zarr_path, target_lat, target_lon,
                 role_to_param, sigma_floor=0.5):
        import xarray as xr
        self.ds = xr.open_zarr(zarr_path)
        self.target_lat = np.asarray(target_lat, dtype=float)
        self.target_lon = np.asarray(target_lon, dtype=float)
        self.role_to_param = dict(role_to_param)
        self.sigma_floor = sigma_floor
        self._lat_name = next((n for n in ("lat", "latitude")
                               if n in self.ds.coords), "lat")
        self._lon_name = next((n for n in ("lon", "longitude")
                               if n in self.ds.coords), "lon")
        self._max_doy = int(self.ds.doy.max().values)
        self._cache: Dict[Tuple[str, int], Tuple] = {}

    def _regrid(self, da2d):
        """Bilinearly interpolate a (lat, lon) DataArray onto the target grid,
        wrapping longitude so the eastmost target column is not NaN."""
        lon = self._lon_name
        # append a wrapped 360-deg column = the 0-deg column
        wrap = da2d.isel({lon: 0}).assign_coords({lon: float(
            da2d[lon].values[0]) + 360.0})
        da_w = __import__("xarray").concat([da2d, wrap], dim=lon)
        out = da_w.interp({self._lat_name: self.target_lat,
                           lon: self.target_lon},
                          method="linear")
        return np.asarray(out.values, dtype=float)

    def fields(self, role: str, init_time: str):
        param = self.role_to_param.get(role)
        if param is None:
            return None, None
        doy = datetime.fromisoformat(init_time).timetuple().tm_yday
        doy = min(doy, self._max_doy)
        key = (param, doy)
        if key not in self._cache:
            sub = self.ds.sel(param=param, doy=doy)
            mean = self._regrid(sub["mean"])
            std = self._regrid(sub["sigma"])
            std = np.clip(std, self.sigma_floor, None)
            self._cache[key] = (mean, std)
        return self._cache[key]


"""
The HMAS evaluation pipeline.

:class:`HMASEvaluator` consumes a :class:`~hmas.stores.ForecastStore` and a
:class:`~hmas.stores.VerificationSource`, computes the six HMAS sub-metrics per
(model, init time), averages them across init times, and forms the composite
HMAS at one or more forecast horizons. The result is a :class:`HMASResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config, metrics
from .config import VariableRoles
from .spectral import compute_power_spectrum_2d, sfi_from_spectra, eff_res_from_spectra
from .stats import (align_grids, compute_weighted_rmse, global_kinetic_energy,
                    make_lat_weights)
from .stores import ForecastStore, VerificationSource

SubScores = Dict[str, float]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class HMASResult:
    """Outcome of an HMAS evaluation.

    Attributes
    ----------
    per_init:
        ``{model: {init_time: {day: {metric: value}}}}`` -- raw sub-metrics
        before averaging.
    aggregated:
        ``{model: {day: {metric: value, "HMAS": value, "n_inits": int}}}`` --
        sub-metrics averaged across init times plus the composite HMAS.
    eval_days, weights:
        The horizons and HMAS weights used.
    """

    per_init: Dict[str, Dict[str, Dict[int, SubScores]]]
    aggregated: Dict[str, Dict[int, SubScores]]
    eval_days: Tuple[int, ...]
    weights: Dict[str, float]

    def scores(self, day: int) -> Dict[str, SubScores]:
        """Per-model aggregated sub-metrics + HMAS at a given day."""
        return {m: d[day] for m, d in self.aggregated.items() if day in d}

    def ranking(self, day: int) -> List[Tuple[str, float]]:
        """Models sorted by HMAS (best first) at ``day``."""
        rows = [(m, d[day].get("HMAS", float("nan")))
                for m, d in self.aggregated.items() if day in d]
        return sorted(rows, key=lambda kv: (np.isnan(kv[1]), -kv[1]
                                            if not np.isnan(kv[1]) else 0))

    def table(self, day: int):
        """Return a pandas DataFrame (rows = models, sorted by HMAS).

        Falls back to a list of dicts if pandas is unavailable.
        """
        cols = list(config.HMAS_METRIC_KEYS) + ["HMAS", "n_inits"]
        ranked = [m for m, _ in self.ranking(day)]
        rows = []
        for m in ranked:
            s = self.aggregated[m][day]
            rows.append({"model": m, **{c: s.get(c, np.nan) for c in cols}})
        try:
            import pandas as pd
            return pd.DataFrame(rows).set_index("model")
        except ImportError:
            return rows

    def to_dict(self) -> dict:
        """JSON-serialisable view of the aggregated scores."""
        return {
            "eval_days": list(self.eval_days),
            "weights": self.weights,
            "aggregated": {
                m: {str(day): {k: (None if (isinstance(v, float) and np.isnan(v))
                                   else v)
                               for k, v in s.items()}
                    for day, s in days.items()}
                for m, days in self.aggregated.items()
            },
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class HMASEvaluator:
    """Compute HMAS from a forecast store and a verification source.

    Parameters
    ----------
    forecast_store:
        Provides forecast fields (any :class:`ForecastStore`).
    verification:
        Provides truth and optional climatology (any
        :class:`VerificationSource`).
    roles:
        Maps the user's variable names to physical roles. Defaults to the
        paper's u500 / z500 / t2m / t850 choices.
    weights:
        HMAS composite weights; defaults to :data:`config.DEFAULT_HMAS_WEIGHTS`.
    lead_tolerance:
        Maximum |forecast - target| lead-time mismatch (hours) tolerated when
        matching to a requested horizon. Default 12.
    """

    def __init__(self,
                 forecast_store: ForecastStore,
                 verification: VerificationSource,
                 roles: Optional[VariableRoles] = None,
                 weights: Optional[Dict[str, float]] = None,
                 lead_tolerance: float = 12.0,
                 verbose: bool = True):
        self.store = forecast_store
        self.verif = verification
        self.roles = roles or VariableRoles()
        self.weights = dict(weights or config.DEFAULT_HMAS_WEIGHTS)
        self.tol = lead_tolerance
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            import sys
            print(msg, flush=True, file=sys.stderr)

    # -- internal helpers ---------------------------------------------------
    def _lats_for(self, model: str, init_time: str, nlat: int) -> np.ndarray:
        lats = self.store.latitudes(model, init_time)
        if lats is None:
            lats = self.verif.latitudes()
        if lats is None or len(lats) != nlat:
            lats = np.linspace(90, -90, nlat)
        return np.asarray(lats, dtype=float)

    def _aligned_pair(self, model, init_time, variable):
        """Return ``(lead_hours, fcst_cube, truth_cube)`` aligned on common
        leads and spatial grid, or ``(None, None, None)``."""
        lh_f, cube_f = self.store.get_series(model, init_time, variable)
        lh_t, cube_t = self.verif.get_truth(variable, init_time)
        if lh_f is None or lh_t is None:
            return None, None, None
        # Match each forecast lead to the nearest truth lead within tolerance.
        f_idx, t_idx = [], []
        for i, h in enumerate(lh_f):
            j = int(np.argmin(np.abs(lh_t - h)))
            if np.abs(lh_t[j] - h) <= self.tol:
                f_idx.append(i)
                t_idx.append(j)
        if not f_idx:
            return None, None, None
        fcst = cube_f[f_idx]
        truth = cube_t[t_idx]
        # spatial alignment
        nlat = min(fcst.shape[1], truth.shape[1])
        nlon = min(fcst.shape[2], truth.shape[2])
        return (lh_f[f_idx], fcst[:, :nlat, :nlon], truth[:, :nlat, :nlon])

    def _evaluate_one_init(self, model: str, init_time: str,
                           eval_days: Sequence[int]) -> Dict[int, SubScores]:
        r = self.roles
        tol = self.tol
        out: Dict[int, SubScores] = {day: {} for day in eval_days}

        # ---- tau_d : error-doubling time from the error-growth variable ----
        tau_d = np.nan
        lh, fcst, truth = self._aligned_pair(model, init_time, r.error_growth)
        if lh is not None:
            lat_w = make_lat_weights(fcst.shape[1])[0]
            rmse = np.array([compute_weighted_rmse(fcst[i], truth[i], lat_w)
                             for i in range(len(lh))])
            good = np.isfinite(rmse)
            if good.sum() >= 3:
                tau_d = metrics.error_doubling_time(rmse[good], lh[good])

        # ---- SFI + eff_res : per-lead spectra of the spectral variable -----
        sfi_series: Dict[float, Tuple[float, float]] = {}
        lh, fcst, truth = self._aligned_pair(model, init_time, r.spectral)
        if lh is not None:
            for i, h in enumerate(lh):
                k_f, p_f = compute_power_spectrum_2d(fcst[i])
                k_a, p_a = compute_power_spectrum_2d(truth[i])
                sfi_series[float(h)] = (
                    sfi_from_spectra(k_f, p_f, k_a, p_a),
                    eff_res_from_spectra(k_f, p_f, k_a, p_a),
                )

        # ---- ASI : drift of the global KE series (wind components) ---------
        asi = np.nan
        lh_u, fcst_u, _ = self._aligned_pair(model, init_time, r.wind_u)
        lh_v, fcst_v, _ = self._aligned_pair(model, init_time, r.wind_v)
        if lh_u is not None and lh_v is not None:
            n = min(len(lh_u), len(lh_v))
            lats = self._lats_for(model, init_time, fcst_u.shape[1])
            ke = np.array([global_kinetic_energy(fcst_u[i], fcst_v[i], lats)
                           for i in range(n)])
            good = np.isfinite(ke) & (ke > 0)
            if good.sum() >= 3:
                asi = metrics.autoregressive_stability(ke[good], lh_u[:n][good])

        # ---- PCS : per-lead dynamical balance ------------------------------
        pcs_series: Dict[float, float] = {}
        snap_hours = sorted({day * 24 for day in eval_days})
        for h in snap_hours:
            u = self.store.get_field(model, init_time, r.wind_u, h, tol)
            v = self.store.get_field(model, init_time, r.wind_v, h, tol)
            z = self.store.get_field(model, init_time, r.geopotential, h, tol)
            if u is None or v is None or z is None:
                continue
            t = (None if r.temperature is None else
                 self.store.get_field(model, init_time, r.temperature, h, tol))
            lats = self._lats_for(model, init_time, u.shape[0])
            pcs, _ = metrics.physical_consistency_score(u, v, z, lats, t_field=t)
            pcs_series[float(h)] = pcs

        # ---- EES : per-lead extreme skill on the extreme variable ----------
        ees_series: Dict[float, float] = {}
        clim_m = self.verif.get_climatology_mean(r.extreme, init_time)
        clim_s = self.verif.get_climatology_std(r.extreme, init_time)
        if clim_m is not None and clim_s is not None:
            for h in snap_hours:
                f = self.store.get_field(model, init_time, r.extreme, h, tol)
                lh_t, cube_t = self.verif.get_truth(r.extreme, init_time)
                if f is None or lh_t is None:
                    continue
                j = int(np.argmin(np.abs(lh_t - h)))
                if np.abs(lh_t[j] - h) > tol:
                    continue
                fa, ta = align_grids(f, cube_t[j])
                cm, _ = align_grids(clim_m, ta)
                cs, _ = align_grids(clim_s, ta)
                ees, _, _ = metrics.extreme_event_skill(fa, ta, cm, cs)
                ees_series[float(h)] = ees

        # ---- assemble per eval day -----------------------------------------
        def nearest(series: Dict[float, object], target: float):
            if not series:
                return None
            key = min(series, key=lambda k: abs(k - target))
            return series[key] if abs(key - target) <= tol else None

        for day in eval_days:
            target = day * 24
            scores: SubScores = {}
            sp = nearest(sfi_series, target)
            if sp is not None:
                if np.isfinite(sp[0]):
                    scores["sfi"] = sp[0]
                if np.isfinite(sp[1]):
                    scores["eff_res"] = sp[1]
            if np.isfinite(tau_d):
                scores["tau_d"] = float(tau_d)
            ees = nearest(ees_series, target)
            if ees is not None and np.isfinite(ees):
                scores["ees"] = float(ees)
            pcs = nearest(pcs_series, target)
            if pcs is not None and np.isfinite(pcs):
                scores["pcs"] = float(pcs)
            if np.isfinite(asi):
                scores["asi"] = float(asi)
            out[day] = scores
        return out

    # -- public API ---------------------------------------------------------
    def evaluate(self,
                 models: Optional[Sequence[str]] = None,
                 init_times: Optional[Sequence[str]] = None,
                 eval_days: Sequence[int] = config.DEFAULT_EVAL_DAYS
                 ) -> HMASResult:
        """Run the full evaluation.

        Parameters
        ----------
        models:
            Models to score. Defaults to every model in the store.
        init_times:
            Init times to average over. Defaults to each model's available
            inits (intersected per model).
        eval_days:
            Forecast horizons (days) at which to report HMAS.

        Returns
        -------
        HMASResult
        """
        models = list(models) if models is not None else self.store.list_models()
        eval_days = tuple(int(d) for d in eval_days)

        per_init: Dict[str, Dict[str, Dict[int, SubScores]]] = {}
        aggregated: Dict[str, Dict[int, SubScores]] = {}

        import time
        t_start = time.time()
        self._log(f"[hmas] evaluating {len(models)} model(s) "
                  f"at days {list(eval_days)}")

        for mi, model in enumerate(models, 1):
            inits = (list(init_times) if init_times is not None
                     else self.store.init_times(model))
            # ISO-8601 timestamps sort lexicographically as chronological;
            # keep a deterministic, time-ordered sequence regardless of the
            # order the store returns them in.
            inits = sorted(inits)
            per_init[model] = {}
            self._log(f"[hmas] ({mi}/{len(models)}) {model}: "
                      f"{len(inits)} init(s)")
            for ii, it in enumerate(inits, 1):
                t0 = time.time()
                per_init[model][it] = self._evaluate_one_init(
                    model, it, eval_days)
                self._log(f"[hmas]   {model} init {ii}/{len(inits)} {it} "
                          f"({time.time() - t0:.1f}s, "
                          f"elapsed {time.time() - t_start:.0f}s)")

            # Average sub-metrics across inits, then compose HMAS.
            aggregated[model] = {}
            for day in eval_days:
                accum: Dict[str, List[float]] = {}
                for it in per_init[model]:
                    for k, v in per_init[model][it][day].items():
                        if v is not None and np.isfinite(v):
                            accum.setdefault(k, []).append(v)
                avg = {k: float(np.mean(vals)) for k, vals in accum.items()}
                if avg:
                    avg["HMAS"] = metrics.holistic_score(
                        avg.get("sfi", np.nan), avg.get("eff_res", np.nan),
                        avg.get("tau_d", np.nan), avg.get("ees", np.nan),
                        avg.get("pcs", np.nan), avg.get("asi", np.nan),
                        weights=self.weights)
                    avg["n_inits"] = (max(len(v) for v in accum.values())
                                      if accum else 0)
                aggregated[model][day] = avg

        return HMASResult(per_init=per_init, aggregated=aggregated,
                          eval_days=eval_days, weights=self.weights)

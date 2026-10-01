#!/usr/bin/env python
"""Full HMAS evaluation.

Computes the complete HMAS bag of metrics for one or more forecast models:
  * per-model composite HMAS + all six sub-metrics (SFI, effective resolution,
    error-doubling time, EES, PCS, ASI), verified against ERA5, with the t2m
    climatology used to define extremes for EES;
  * the cross-model Error Consensus Ratio (ECR), if >= 3 models are selected.

"""
import os
# HDF5 file locking off -- required on many /shared (Lustre/GPFS/NFS) mounts and
# must be set BEFORE netCDF4/xarray import it.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import json
import numpy as np

from hmas import HMASEvaluator, NetCDFForecastStore, VariableRoles
from hmas.consensus import error_consensus_ratio
from hmas.adapters.era5 import ERA5VerificationSource

# ----------------------------------------------------------------- config
FORECAST_DIR = "/path/to/forecasts"
VERIF_DIR    = "/path/to/era5_verif"
CLIM_DIR     = "/path/to/era5_data"       # holds the t2m dayofyear clim
MODELS       = ["aifsens", "graphcast"]   # set as you like; >=3 enables ECR
EVAL_DAYS    = [3, 5, 15]
LEAD_HOURS   = np.arange(6, 367, 6)       # the 6..366 h forecast grid
ECR_VARS     = ("z500", "t2m", "t850")
OUT_JSON     = "hmas_results.json"

# ----------------------------------------------------------------- build
store = NetCDFForecastStore(FORECAST_DIR)
print("models in store:", store.list_models())

verif = ERA5VerificationSource(
    verif_dir=VERIF_DIR,
    clim_dir=CLIM_DIR,           # built-in dayofyear reader picks up t2m clim
    lead_hours=LEAD_HOURS,
)

# climatology sanity check -- EES needs both mean and sigma for t2m
probe_init = store.init_times(MODELS[0])[0]
cm = verif.get_climatology_mean("t2m", probe_init)
cs = verif.get_climatology_std("t2m", probe_init)
print("t2m clim  mean:", None if cm is None else cm.shape,
      "| sigma:", None if cs is None else cs.shape)
if cm is None or cs is None:
    print("WARNING: no t2m climatology resolved from", CLIM_DIR,
          "-- EES will be skipped and HMAS renormalized over the other five.")

# ----------------------------------------------------------------- HMAS
ev = HMASEvaluator(store, verif, roles=VariableRoles())   # extreme role = t2m
result = ev.evaluate(models=MODELS, eval_days=EVAL_DAYS)

for day in EVAL_DAYS:
    print(f"\n===== HMAS day {day} {'='*40}")
    # table() -> pandas DataFrame: rows=models, cols=sub-metrics + HMAS
    print(result.table(day=day))
    print("ranking:", result.ranking(day=day))

# ----------------------------------------------------------------- ECR (>=3)
ecr_summary = {}      # {var: {day: mean_ecr}}
ecr_raw = {}          # {var: [per-init dicts]}
if len(MODELS) >= 3:
    inits = store.init_times(MODELS[0])
    for var in ECR_VARS:
        per_init = []
        for it in inits:
            r = error_consensus_ratio(store, verif, MODELS, it, var,
                                      min_models=3)
            if r is not None:
                per_init.append(r)
        ecr_raw[var] = per_init
        # mean ECR across inits at each eval day (nearest lead)
        day_means = {}
        for day in EVAL_DAYS:
            target = day * 24
            vals = []
            for r in per_init:
                lh = np.asarray(r["lead_hours"], dtype=float)
                if len(lh) == 0:
                    continue
                j = int(np.argmin(np.abs(lh - target)))
                if abs(lh[j] - target) <= 12 and np.isfinite(r["ecr"][j]):
                    vals.append(float(r["ecr"][j]))
            day_means[day] = float(np.mean(vals)) if vals else None
        ecr_summary[var] = day_means
        print(f"\nECR[{var}] over {len(per_init)} inits:",
              {d: (None if v is None else round(v, 3))
               for d, v in day_means.items()})
else:
    print(f"\nECR needs >=3 models; have {len(MODELS)}, skipping.")

# ----------------------------------------------------------------- save
def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return None

payload = {
    "models": MODELS,
    "eval_days": EVAL_DAYS,
    "hmas": result.to_dict(),
    "ecr_summary": ecr_summary,
    "ecr_raw": ecr_raw,
}
with open(OUT_JSON, "w") as f:
    json.dump(payload, f, indent=2, default=_jsonable)
print("\nwrote", OUT_JSON)

# ------------------------------------------------------- canonical artifacts
# Write the figure-ready JSONs (holistic_scores_day{N}.json, hmas_correlation
# .json) into RESULTS_DIR, then regenerate every figure whose artifact exists.
# Drop the curve artifacts (spectral_curves.json, rmse_curves.json) into the
# same RESULTS_DIR -- via hmas.figures.curves exporters on the precomputed
# cache -- to additionally produce Figs 1-3.
from hmas import figures as F

RESULTS_DIR = "hmas_results"
FIG_DIR = "figures"
F.write_hmas_artifacts(result, RESULTS_DIR)
written = F.generate_all(RESULTS_DIR, out_dir=FIG_DIR, days=EVAL_DAYS)
print(f"generated {len(written)} figures into {FIG_DIR}/")

verif.close()

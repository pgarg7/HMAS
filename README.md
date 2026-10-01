# hmas — Holistic Model Assessment for AI weather forecasts

`hmas` is an **engine-agnostic** implementation of the diagnostic framework
behind the **Holistic Model Assessment Score (HMAS)**. It takes *any* forecast
store — yours, not tied to any particular inference engine — verifies it against
ERA5 (and a climatology), and computes the six physically grounded sub-metrics
and their composite, plus the cross-model Error Consensus Ratio. It also
regenerates the paper's figures from saved metric artifacts.

| Metric | Symbol | What it measures |
|---|---|---|
| Spectral Fidelity Index | SFI | agreement of the energy spectrum with truth |
| Effective resolution | ℓ_eff | finest scale resolved within an acceptable energy band |
| Error-doubling time | τ_d | exponential error-growth rate (predictability) |
| Extreme Event Skill | EES | conditional skill over >2σ tail exceedances |
| Physical Consistency Score | PCS | geostrophic / non-divergence / thermal-wind / hydrostatic balance |
| Autoregressive Stability Index | ASI | drift of global kinetic energy over the rollout |
| **Holistic Model Assessment Score** | **HMAS** | weighted composite of the six |
| Error Consensus Ratio | ECR | shared vs model-specific error across a model ensemble |

## Install

`hmas` is installed from a checkout of this repository. The bundled conda
environment provides the dependencies (`numpy`, `scipy`, `pandas`, `xarray`,
`netCDF4`, `h5netcdf`, `zarr`, `matplotlib`, plus `pytest`) and puts `hmas`
into the environment in editable mode:

```bash
conda env create -f environment.yml
conda activate hmas
```

Verify the layout imported cleanly (run this from *outside* the source tree):

```bash
python -c "import hmas; print(hmas.__version__); \
from hmas.adapters.era5 import ERA5VerificationSource; \
from hmas import figures; print('ok')"
```

## Quick start

You supply forecasts via a `ForecastStore` and truth via a
`VerificationSource`. The simplest are the in-memory array versions:

```python
import numpy as np
import hmas

# forecast_data[model][init_time][variable] = (lead_hours[N], cube[N,lat,lon])
store = hmas.ArrayForecastStore(forecast_data, latitudes=lats)

# truth[init_time][variable] = (lead_hours, cube)
# climatology mean/std (for EES) keyed the same way, optional
verif = hmas.ArrayVerification(truth, climatology_mean=cm, climatology_std=cs,
                               latitudes=lats)

result = hmas.HMASEvaluator(store, verif).evaluate(eval_days=[3, 5, 15])

print(result.table(day=5))          # pandas DataFrame, sorted by HMAS
print(result.ranking(3))            # [(model, HMAS), ...] best first
result.to_dict()                    # JSON-serialisable
```

The evaluator logs progress to stderr (per model, per init, with timing); pass
`HMASEvaluator(..., verbose=False)` to silence it.

### Mapping your variable names

HMAS sub-metrics need specific physical fields. If your variables aren't named
`u500/v500/z500/t2m/t850`, declare the mapping once:

```python
from hmas import VariableRoles

roles = VariableRoles(
    spectral="U_500hPa",       # SFI, effective resolution
    error_growth="Z_500hPa",   # error-doubling time
    extreme="T_2m",            # EES (needs climatology mean + std)
    wind_u="U_500hPa", wind_v="V_500hPa",
    geopotential="Z_500hPa",
    temperature="T_850hPa",    # set None to skip thermal-wind/hydrostatic PCS
)
hmas.HMASEvaluator(store, verif, roles=roles).evaluate()
```

### Custom HMAS weights

```python
weights = {"sfi": .25, "eff_res": .15, "tau_d": .10,
           "ees": .15, "pcs": .15, "asi": .20}
hmas.HMASEvaluator(store, verif, weights=weights).evaluate()
```

## Reading from NetCDF

If your forecasts live in `{root}/{model}/{YYYYMMDDTHHMMSS}*.nc` files with
`(lead_time, lat, lon)` arrays:

```python
store = hmas.NetCDFForecastStore("/path/to/forecasts")
```

`z500`-type variables are auto-converted from geopotential to geopotential
height when their magnitude indicates it.

## ERA5 verification adapter (`hmas.adapters.era5`)

`ERA5VerificationSource` reads per-init ERA5 verification files
(`era5_verif_YYYYMMDD.nc`, a `valid_time`/`time` dimension) and a day-of-year
t2m climatology, with no dependency on any inference engine. It resolves
variables by short name (`z500`, `t2m`, ...) or CDS name + pressure level,
auto-detects z500 geopotential vs height, matches to the nearest valid time
within 6 h, and floors the climatology σ at 0.5 K.

```python
from hmas import HMASEvaluator, NetCDFForecastStore, VariableRoles
from hmas.adapters.era5 import ERA5VerificationSource
import numpy as np

store = NetCDFForecastStore("/data/results/forecasts")
verif = ERA5VerificationSource(
    verif_dir="/data/era5_verif",      # era5_verif_YYYYMMDD.nc files
    clim_dir="/data/era5_data",        # holds the dayofyear t2m climatology
    lead_hours=np.arange(6, 367, 6),   # the forecast lead grid (hours)
)
result = HMASEvaluator(store, verif).evaluate(models=["aifsens", "graphcast"],
                                              eval_days=[3, 5, 15])
```

A single unreadable verification file is logged and skipped (the affected init
contributes fewer sub-metrics) rather than aborting the run. On clustered
filesystems set `HDF5_USE_FILE_LOCKING=FALSE` before the Python process starts
(see *Notes* below).

### Coarse climatology (ChaosBench zarr)

`ChaosBenchClimatology` provides a climatology from a ChaosBench spatial zarr
(dims `doy, param, lat, lon`; data vars `mean`, `sigma`). It selects a `param`
code (e.g. `t-850`), picks the init's day-of-year, and **regrids** `mean`/`sigma`
onto the forecast grid (the evaluator aligns by truncation, not interpolation,
so coarse climatologies must be regridded first):

```python
from hmas.adapters.era5 import ERA5VerificationSource, ChaosBenchClimatology

clim = ChaosBenchClimatology(
    "/data/climatology_era5_spatial.zarr",
    target_lat=lats, target_lon=lons,        # forecast grid
    role_to_param={"t2m": "t-1000", "t850": "t-850"},
)
verif = ERA5VerificationSource(verif_dir="/data/era5_verif", climatology=clim)
```

## Figure generation (`hmas.figures`)

Figures regenerate from saved metric artifacts — no ERA5, cache, or inference
engine needed — so reviewers reproduce them from the committed JSONs alone.

**Directory-driven (recommended).** Run the metric computation so it writes
artifacts into a results directory under canonical filenames, then point
`generate_all` at it; it renders every figure whose input is present and skips
the rest:

```python
from hmas import figures as F

# from a package HMAS run: write the figure-ready artifacts first
F.write_hmas_artifacts(result, "hmas_results")     # holistic_scores_day{N}.json, hmas_correlation.json
F.generate_all("hmas_results", out_dir="figures")  # renders all available figures
```

`generate_all` reads these canonical filenames:

| Input artifact | Figure(s) |
|---|---|
| `spectral_curves.json` | Fig 1 power spectra |
| `spectral_ratio_curves.json` | Fig 2 spectral-energy ratio (u500, z500) |
| `rmse_curves.json` | Fig 3 RMSE vs lead time |
| `acc_metrics.json` | Fig 4 ACC vs lead time |
| `holistic_scores_day{N}.json` | Fig 11 radar + parallel, per day |
| `hmas_correlation.json` | Fig 14 sub-metric cross-correlation |

(Additional paper figures are being added to the registry in figure-number
order; each is one entry of `output_stem -> (input_filename, render_fn)`.)

**Individual figures.** Every function also takes a path or a loaded dict and
returns a matplotlib `Figure`:

```python
from hmas.figures import holistic_radar, hmas_correlation
from hmas.figures.style import savefig

fig = holistic_radar("hmas_results/holistic_scores_day5.json", eval_day=5)
savefig(fig, "fig11_holistic_radar_day5")          # writes .pdf + .png
```

**Curve artifacts.** Figs 1–3 plot curves that live in the precomputed analysis
cache, not in the scalar JSONs. Export them once (where the cache lives) into the
same results directory, after which `generate_all` picks them up:

```python
from hmas.figures import curves
curves.export_spectral_curves(cache, "hmas_results/spectral_curves.json")
curves.export_spectral_ratio_curves(cache, "hmas_results/spectral_ratio_curves.json")
curves.export_rmse_curves(cache, "hmas_results/rmse_curves.json")
```

Figures that are computed by a separate analysis pipeline (ACC, scorecard,
regional RMSE, etc.) just need their JSONs dropped into the same results
directory under the canonical names; `generate_all` renders them automatically.

## Bring your own store

Implement three methods and the rest is handled for you:

```python
import numpy as np
from hmas import ForecastStore

class MyStore(ForecastStore):
    def list_models(self):
        return ["mymodel"]

    def init_times(self, model):
        return ["2023-01-15T00:00:00"]

    def get_series(self, model, init_time, variable):
        # return (lead_hours[N], cube[N, nlat, nlon]) ordered North->South,
        # or (None, None) if unavailable
        ...
        return lead_hours, cube
```

`get_field(...)` (nearest-lead lookup) and `latitudes(...)` have sensible
defaults derived from `get_series`. A `VerificationSource` similarly needs only
`get_truth(variable, init_time)`, plus optional `get_climatology_mean/std` for
EES.

## Using the metric functions directly

Every metric is a pure function over numpy fields:

```python
from hmas import metrics

sfi  = metrics.spectral_fidelity_index(fcst_field, truth_field)
taud = metrics.error_doubling_time(rmse_series, lead_hours)
pcs, sub = metrics.physical_consistency_score(u, v, z, lats, t_field=t)
asi  = metrics.autoregressive_stability(ke_series, lead_hours)
hmas_score = metrics.holistic_score(sfi, eff_res, taud, ees, pcs, asi)
```

## Notes & conventions

- Grids are assumed regular lat-lon ordered **North (+90) → South (−90)**.
  Pass an explicit latitude vector through the store/verification if yours
  differs.
- Forecasts and truth must be in **consistent units**. `z500` geopotential vs
  geopotential height is auto-detected inside PCS/KE and on NetCDF/ERA5 read.
- Missing sub-metrics (`nan`) are dropped from the HMAS composite and the
  remaining weights are renormalised, so partial inputs still yield a score.
- EES needs a climatology for the `extreme` variable (mean + σ); without it EES
  is skipped and HMAS renormalises over the other five.
- ECR requires at least three distinct models.
- **Clustered filesystems (Lustre/GPFS/NFS):** set
  `HDF5_USE_FILE_LOCKING=FALSE` *before* the Python process starts (it is read
  by libhdf5 at load time). For conda: `conda env config vars set
  HDF5_USE_FILE_LOCKING=FALSE -n <env>`, then reactivate and restart the
  kernel. In VS Code, a `.env` file with that line at the workspace root also
  works. A successful `ncdump -h file.nc` while Python raises `Errno -101` is
  the signature of this locking issue, not a corrupt file.

## License

MIT.

"""Tests for the hmas package."""

from __future__ import annotations

import numpy as np
import pytest

import hmas
from hmas import metrics
from synthetic import make_dataset, truth_fields


# ---------------------------------------------------------------------------
# Metric-level tests
# ---------------------------------------------------------------------------
def test_sfi_perfect_is_one():
    truth, _ = truth_fields()
    z = truth["z500"][1][0]
    assert metrics.spectral_fidelity_index(z, z) == pytest.approx(1.0, abs=1e-9)


def test_sfi_degraded_is_lower():
    truth, _ = truth_fields()
    z = truth["z500"][1][0]
    rng = np.random.default_rng(0)
    noisy = z + rng.normal(0, 50, z.shape)
    assert metrics.spectral_fidelity_index(noisy, z) < 1.0


def test_effective_resolution_bounds():
    truth, _ = truth_fields()
    z = truth["z500"][1][0]
    er = metrics.effective_resolution(z, z)
    assert 0.0 <= er <= 1.0


def test_error_doubling_time_exponential():
    hours = np.arange(0, 240, 24.0)
    # RMSE doubling exactly every 48 h -> tau_d = 48 h -> normalized 1.0
    rmse = 1.0 * 2 ** (hours / 48.0)
    td = metrics.error_doubling_time(rmse, hours)
    assert td == pytest.approx(1.0, abs=1e-6)


def test_error_doubling_time_fast_growth_below_one():
    hours = np.arange(0, 240, 24.0)
    rmse = 1.0 * 2 ** (hours / 24.0)  # doubles every 24 h -> tau_d/48 = 0.5
    td = metrics.error_doubling_time(rmse, hours)
    assert td == pytest.approx(0.5, abs=1e-6)


def test_ees_perfect_everywhere_is_nan():
    # A flawless forecast has zero unconditional RMSE -> EES is 0/0 -> nan.
    truth, _ = truth_fields()
    t = truth["t2m"][1][0]
    cm = t * 0 + t.mean()
    cs = t * 0 + 3.0
    ees, _, crmse = metrics.extreme_event_skill(t, t, cm, cs)
    assert np.isnan(ees) and crmse == pytest.approx(0.0, abs=1e-9)


def test_ees_perfect_on_extremes_is_one():
    # Forecast nails the extreme points but errs elsewhere -> EES == 1.
    truth, _ = truth_fields()
    t = truth["t2m"][1][0]
    cm = t * 0 + t.mean()
    cs = t * 0 + 3.0
    threshold = cm + 2.0 * cs
    fcst = t.copy()
    away = t <= threshold
    fcst[away] += 5.0  # error only away from the warm tail
    ees, _, _ = metrics.extreme_event_skill(fcst, t, cm, cs)
    assert ees == pytest.approx(1.0, abs=1e-9)


def test_ees_ranks_extreme_skill():
    # Good-on-extremes scores higher than bad-on-extremes.
    truth, _ = truth_fields()
    t = truth["t2m"][1][0]
    cm = t * 0 + t.mean()
    cs = t * 0 + 3.0
    threshold = cm + 2.0 * cs
    hot = t > threshold
    rng = np.random.default_rng(0)
    base = t + rng.normal(0, 1.0, t.shape)
    good = base.copy(); good[hot] = t[hot]            # perfect on extremes
    bad = base.copy(); bad[hot] = t[hot] + 8.0         # poor on extremes
    ees_good = metrics.extreme_event_skill(good, t, cm, cs)[0]
    ees_bad = metrics.extreme_event_skill(bad, t, cm, cs)[0]
    assert ees_good > ees_bad


def test_pcs_balanced_truth_high():
    truth, lats = truth_fields()
    u, v, z, t = (truth[k][1][0] for k in ("u500", "v500", "z500", "t850"))
    pcs, sub = metrics.physical_consistency_score(u, v, z, lats, t_field=t)
    # Truth winds are geostrophic by construction -> strong geostrophic score.
    assert 0.0 <= pcs <= 1.0
    assert sub["geostrophic"] > 0.5


def test_asi_stable_vs_drifting():
    hours = np.arange(0, 360, 24.0)
    stable = np.full_like(hours, 100.0)
    drifting = 100.0 * np.exp(0.01 * hours)
    asi_stable = metrics.autoregressive_stability(stable, hours)
    asi_drift = metrics.autoregressive_stability(drifting, hours)
    assert asi_stable == pytest.approx(1.0, abs=1e-6)
    assert asi_drift < asi_stable


def test_holistic_score_renormalizes_on_nan():
    full = metrics.holistic_score(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
    assert full == pytest.approx(0.8, abs=1e-9)
    partial = metrics.holistic_score(0.8, 0.8, 0.8, np.nan, np.nan, 0.8)
    assert partial == pytest.approx(0.8, abs=1e-9)  # constant inputs -> 0.8
    assert np.isnan(metrics.holistic_score(*([np.nan] * 6)))


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------
def test_array_store_roundtrip():
    forecast, truth, cm, cs, lats = make_dataset()
    store = hmas.ArrayForecastStore(forecast, latitudes=lats)
    assert store.list_models() == ["good_model", "poor_model"]
    lh, cube = store.get_series("good_model", "2023-01-15T00:00:00", "z500")
    assert cube.ndim == 3
    field = store.get_field("good_model", "2023-01-15T00:00:00", "z500", 72.0)
    assert field.shape == cube.shape[1:]
    # 1000 h is far from any available lead (max 360) -> out of tolerance
    assert store.get_field("good_model", "2023-01-15T00:00:00", "z500",
                           1000.0, tolerance=12) is None


def test_verification_climatology():
    _, truth, cm, cs, lats = make_dataset()
    verif = hmas.ArrayVerification(truth, cm, cs, latitudes=lats)
    assert verif.get_climatology_mean("t2m", "2023-01-15T00:00:00") is not None
    assert verif.get_climatology_std("t2m", "2023-01-15T00:00:00") is not None


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------
def test_pipeline_end_to_end():
    forecast, truth, cm, cs, lats = make_dataset()
    store = hmas.ArrayForecastStore(forecast, latitudes=lats)
    verif = hmas.ArrayVerification(truth, cm, cs, latitudes=lats)
    result = hmas.HMASEvaluator(store, verif).evaluate(eval_days=[3, 5, 15])

    for model in ("good_model", "poor_model"):
        for day in (3, 5, 15):
            s = result.aggregated[model][day]
            assert "HMAS" in s
            assert 0.0 <= s["HMAS"] <= 1.0
            # every sub-metric present and in range
            for k in hmas.HMAS_METRIC_KEYS:
                if k in s:
                    assert 0.0 <= s[k] <= 1.0


def test_good_model_beats_poor_model():
    forecast, truth, cm, cs, lats = make_dataset()
    store = hmas.ArrayForecastStore(forecast, latitudes=lats)
    verif = hmas.ArrayVerification(truth, cm, cs, latitudes=lats)
    result = hmas.HMASEvaluator(store, verif).evaluate(eval_days=[5])
    good = result.aggregated["good_model"][5]["HMAS"]
    poor = result.aggregated["poor_model"][5]["HMAS"]
    assert good > poor


def test_result_ranking_and_dict():
    forecast, truth, cm, cs, lats = make_dataset()
    store = hmas.ArrayForecastStore(forecast, latitudes=lats)
    verif = hmas.ArrayVerification(truth, cm, cs, latitudes=lats)
    result = hmas.HMASEvaluator(store, verif).evaluate(eval_days=[5])
    ranking = result.ranking(5)
    assert ranking[0][0] == "good_model"  # best first
    d = result.to_dict()
    assert "aggregated" in d and "good_model" in d["aggregated"]


def test_custom_weights():
    forecast, truth, cm, cs, lats = make_dataset()
    store = hmas.ArrayForecastStore(forecast, latitudes=lats)
    verif = hmas.ArrayVerification(truth, cm, cs, latitudes=lats)
    w = {"sfi": 1.0, "eff_res": 0, "tau_d": 0, "ees": 0, "pcs": 0, "asi": 0}
    result = hmas.HMASEvaluator(store, verif, weights=w).evaluate(eval_days=[5])
    s = result.aggregated["good_model"][5]
    assert s["HMAS"] == pytest.approx(s["sfi"], abs=1e-9)


# ---------------------------------------------------------------------------
# Consensus tests
# ---------------------------------------------------------------------------
def test_ecr_requires_three_models():
    forecast, truth, cm, cs, lats = make_dataset()
    store = hmas.ArrayForecastStore(forecast, latitudes=lats)
    verif = hmas.ArrayVerification(truth, cm, cs, latitudes=lats)
    out = hmas.error_consensus_ratio(
        store, verif, ["good_model", "poor_model"],
        "2023-01-15T00:00:00", "z500")
    assert out is None  # only two models


def test_ecr_three_models():
    forecast, truth, cm, cs, lats = make_dataset()
    # add a third model
    from synthetic import degrade
    forecast["mid_model"] = {
        "2023-01-15T00:00:00":
        {v: degrade(truth["2023-01-15T00:00:00"][v], 2, 4.0, 3) for v in
         truth["2023-01-15T00:00:00"]}}
    store = hmas.ArrayForecastStore(forecast, latitudes=lats)
    verif = hmas.ArrayVerification(truth, cm, cs, latitudes=lats)
    out = hmas.error_consensus_ratio(
        store, verif, list(forecast.keys()), "2023-01-15T00:00:00", "z500")
    assert out is not None
    assert out["n_models"] == 3
    assert np.all((out["ecr"][np.isfinite(out["ecr"])] >= 0)
                  & (out["ecr"][np.isfinite(out["ecr"])] <= 1))

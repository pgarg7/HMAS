"""
hmas -- Holistic Model Assessment for AI weather forecasts.

A lightweight, engine-agnostic implementation of the diagnostic framework
behind the Holistic Model Assessment Score (HMAS). It computes six
physically grounded sub-metrics -- Spectral Fidelity Index (SFI), effective
resolution, error-doubling time, Extreme Event Skill (EES), Physical
Consistency Score (PCS) and Autoregressive Stability Index (ASI) -- and the
composite HMAS, plus the cross-model Error Consensus Ratio (ECR).

The core depends only on ``numpy`` (``scipy``/``pandas`` optional). It makes
no reference to any inference engine: you supply forecasts through a
:class:`ForecastStore` and truth through a :class:`VerificationSource`.

Quick start
-----------
>>> from hmas import HMASEvaluator, ArrayForecastStore, ArrayVerification
>>> store = ArrayForecastStore(forecast_data)
>>> verif = ArrayVerification(truth, climatology_mean=cm, climatology_std=cs)
>>> result = HMASEvaluator(store, verif).evaluate(eval_days=[3, 5, 15])
>>> result.table(day=5)
"""

from __future__ import annotations

__version__ = "0.2.3"

from . import config, consensus, metrics, spectral, stats, stores
from .config import (DEFAULT_HMAS_WEIGHTS, HMAS_METRIC_KEYS, VariableRoles)
from .consensus import error_consensus_ratio, pairwise_error_correlation
from .metrics import (MetricsComputer, autoregressive_stability,
                      effective_resolution, error_doubling_time,
                      extreme_event_skill, holistic_score,
                      physical_consistency_score, spectral_fidelity_index)
from .pipeline import HMASEvaluator, HMASResult
from .stores import (ArrayForecastStore, ArrayVerification, ForecastStore,
                     NetCDFForecastStore, VerificationSource)

__all__ = [
    "__version__",
    # high-level
    "HMASEvaluator", "HMASResult",
    # stores
    "ForecastStore", "ArrayForecastStore", "NetCDFForecastStore",
    "VerificationSource", "ArrayVerification",
    # config
    "VariableRoles", "DEFAULT_HMAS_WEIGHTS", "HMAS_METRIC_KEYS",
    # metric functions
    "MetricsComputer", "spectral_fidelity_index", "effective_resolution",
    "error_doubling_time", "extreme_event_skill",
    "physical_consistency_score", "autoregressive_stability",
    "holistic_score",
    # consensus
    "error_consensus_ratio", "pairwise_error_correlation",
    # submodules
    "config", "consensus", "metrics", "spectral", "stats", "stores",
]

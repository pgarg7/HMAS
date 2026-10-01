"""
Configuration: physical constants, variable roles, and default metric weights.

Nothing in this module is tied to any particular inference engine or file
format. A user supplies their own forecast data through a
:class:`hmas.stores.ForecastStore` and tells the evaluator which of their
variables plays each physical *role* via :class:`VariableRoles`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G = 9.80665          # gravitational acceleration, m s^-2
OMEGA = 7.2921e-5    # Earth angular velocity, s^-1
R_EARTH = 6.371e6    # Earth radius, m
R_DRY = 287.05       # dry-air gas constant, J kg^-1 K^-1


# ---------------------------------------------------------------------------
# HMAS composite weights
# ---------------------------------------------------------------------------
# Weights reflect both operational importance and discriminating power across
# a ten-model evaluation. ASI and SFI carry the most weight (energy
# conservation and the loss-function spectral signature are the most
# discriminating); tau_d is demoted because error-doubling time is primarily
# an atmospheric property with limited model discrimination; PCS is promoted
# because physical consistency captures a unique, near-orthogonal dimension.
DEFAULT_HMAS_WEIGHTS: Dict[str, float] = {
    "sfi": 0.20,
    "eff_res": 0.15,
    "tau_d": 0.15,
    "ees": 0.15,
    "pcs": 0.15,
    "asi": 0.20,
}

# The six sub-metric keys, in canonical order.
HMAS_METRIC_KEYS = ("sfi", "eff_res", "tau_d", "ees", "pcs", "asi")

# Pretty labels for tables / plots.
HMAS_METRIC_LABELS = {
    "sfi": "SFI",
    "eff_res": "l_eff",
    "tau_d": "tau_d",
    "ees": "EES",
    "pcs": "PCS",
    "asi": "ASI",
}

# Equal weights for the four PCS dynamical-balance sub-scores: each balance
# constraint is physically fundamental, so unequal weighting would be
# arbitrary.
PCS_BALANCE_WEIGHTS: Dict[str, float] = {
    "geostrophic": 0.25,
    "nondivergence": 0.25,
    "thermal_wind": 0.25,
    "hydrostatic": 0.25,
}

# Lead days (forecast horizon, in days) at which HMAS is reported by default.
DEFAULT_EVAL_DAYS: Sequence[int] = (3, 5, 15)


@dataclass
class VariableRoles:
    """Maps a user's variable names onto the physical roles HMAS requires.

    Each HMAS sub-metric is computed from a specific physical field. Because a
    user's store may name variables anything (``"u_500hPa"``, ``"Z500"``,
    ``"2t"`` ...), the evaluator never hard-codes names; it asks for these
    roles instead.

    Attributes
    ----------
    spectral:
        Variable used for the Spectral Fidelity Index (SFI) and effective
        resolution. The paper uses ``u500`` (500 hPa zonal wind), whose
        spectrum is sensitive to the loss-function signature.
    error_growth:
        Variable whose global RMSE curve is fit to obtain the error-doubling
        time ``tau_d``. The paper uses ``z500``.
    extreme:
        Surface variable for Extreme Event Skill (EES). The paper uses
        ``t2m``. Requires a climatology (mean and std) from the verification
        source.
    wind_u, wind_v:
        Horizontal wind components for the kinetic-energy series (ASI) and
        geostrophic / non-divergence balance (PCS). The paper uses ``u500``,
        ``v500``.
    geopotential:
        Geopotential (height) field for PCS geostrophic / thermal-wind /
        hydrostatic balances. The paper uses ``z500``.
    temperature:
        Optional layer temperature (paper: ``t850``) enabling the thermal-wind
        and hydrostatic PCS sub-scores. If ``None``, PCS uses only the
        geostrophic and non-divergence balances.
    """

    spectral: str = "u500"
    error_growth: str = "z500"
    extreme: str = "t2m"
    wind_u: str = "u500"
    wind_v: str = "v500"
    geopotential: str = "z500"
    temperature: str | None = "t850"

    def required_variables(self) -> list[str]:
        """Unique set of variable names this role mapping references."""
        names = [
            self.spectral,
            self.error_growth,
            self.extreme,
            self.wind_u,
            self.wind_v,
            self.geopotential,
        ]
        if self.temperature is not None:
            names.append(self.temperature)
        # de-duplicate, preserve order
        seen: dict[str, None] = {}
        for n in names:
            seen.setdefault(n, None)
        return list(seen)


# Variables used by the cross-model Error Consensus Ratio (ECR) by default.
DEFAULT_CONSENSUS_VARS: Sequence[str] = ("z500", "t2m", "t850")

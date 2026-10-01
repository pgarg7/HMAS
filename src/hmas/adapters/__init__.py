"""Data-source adapters that bridge specific on-disk layouts to the
engine-agnostic hmas core. These are intentionally opinionated about file
conventions, unlike the generic stores in :mod:`hmas.stores`.
"""
from .era5 import ERA5VerificationSource, ChaosBenchClimatology

__all__ = ["ERA5VerificationSource", "ChaosBenchClimatology"]

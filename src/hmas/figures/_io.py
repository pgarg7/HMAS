"""Small helper to accept either a path to a metrics JSON file or an
already-loaded dict, so every figure function works both from the saved
artifacts (reviewer reproduction) and from in-memory results.
"""
from __future__ import annotations

import json
import os
from typing import Union

DataLike = Union[str, os.PathLike, dict, list]


def load(obj: DataLike):
    """Return a parsed object from a path, or pass through a dict/list."""
    if isinstance(obj, (dict, list)):
        return obj
    with open(obj, "r") as f:
        return json.load(f)

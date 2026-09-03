"""Canonical simulation namespace for the GRAIN analysis method."""

from __future__ import annotations

import suica_sim as _legacy_sim

__path__ = _legacy_sim.__path__

from suica_sim import (  # noqa: E402,F401
    conditional_innovation,
    endpoint_flow_variance,
    load_config,
    run_matrix,
    stationary_mean_variance,
    write_reports,
)

__all__ = [
    "conditional_innovation",
    "endpoint_flow_variance",
    "load_config",
    "run_matrix",
    "stationary_mean_variance",
    "write_reports",
]


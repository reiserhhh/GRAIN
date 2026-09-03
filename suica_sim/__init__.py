"""Legacy simulation namespace for the GRAIN theorem matrix."""

from .algebra import conditional_innovation, endpoint_flow_variance, stationary_mean_variance
from .matrix import load_config, run_matrix, write_reports

__all__ = [
    "conditional_innovation",
    "endpoint_flow_variance",
    "load_config",
    "run_matrix",
    "stationary_mean_variance",
    "write_reports",
]

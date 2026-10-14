"""Finite, auditable research components for GRAIN measurement discovery."""

from .contracts import (
    RECORD_FIELDS,
    TARGET_FIELDS,
    target_contract,
    validate_records,
    validate_split,
)

__all__ = [
    "RECORD_FIELDS",
    "TARGET_FIELDS",
    "target_contract",
    "validate_records",
    "validate_split",
]

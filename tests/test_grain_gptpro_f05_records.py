"""Contract conversion preserves exact observations and filtering."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f00 import validate_record
from grain_research.gptpro_foundation_v3.f05_records import from_common, to_common
from grain_research.gptpro_foundation_v3.f05_recovery import (
    generate_dynamics,
    kalman_loglik,
)


def test_full_filter_roundtrip_and_truth_rejection():
    rows, _ = generate_dynamics(
        "core", 12, {"source": 1, "validation": 0, "application": 0}, 32
    )
    converted = [to_common(row) for row in rows]
    seen = set()
    for row in converted:
        validate_record(row, {row["author_id"]: "source"}, seen)
    back = [from_common(row) for row in converted]
    assert rows == back
    assert kalman_loglik([0.0, 0.5, 0.7], rows, np.log([0.5, 0.5])) == kalman_loglik(
        [0.0, 0.5, 0.7], back, np.log([0.5, 0.5])
    )
    with pytest.raises(ValueError):
        to_common({**rows[0], "truth": 1})

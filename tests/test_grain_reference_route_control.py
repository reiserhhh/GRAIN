#!/usr/bin/env python3
"""Reader provenance must not silently change target or observation membership."""

import pandas as pd
import pytest

from scripts.run_grain_reference_route_control import natural_targets


def test_reader_factorial_targets():
    frame = pd.DataFrame(
        [
            {"world_id": "w", "query": "q", "readout": arm, "score": value}
            for arm, value in (("native", 0.7), ("R_real", 0.6), ("R_synth", 0.5))
        ]
    )
    target = natural_targets(frame).set_index("reference")
    assert target.loc["R_real", "target"] == 0.6
    assert target.loc["R_synth", "target"] == 0.5
    assert target.native_score.tolist() == [0.7, 0.7]
    with pytest.raises(ValueError):
        natural_targets(frame.iloc[:2])
    with pytest.raises(ValueError):
        natural_targets(pd.concat([frame, frame.iloc[:1]]))

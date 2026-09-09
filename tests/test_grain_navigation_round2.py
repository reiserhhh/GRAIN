#!/usr/bin/env python3
"""Signed factorial terms and exact same-person view comparisons."""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_grain_counseling_views import paired_delta
from scripts.analyze_grain_local_route_atlas import factorial_differences
from scripts.analyze_grain_state_opportunity import cross_occasion_links


def test_mixed_rectangle_recovers_interaction_not_main_effects():
    rows = []
    for actor, a in (("self", 0), ("other", 1)):
        for opportunity, b in (("voluntary", 0), ("assigned", 1)):
            rows.append(
                {
                    "query": "q",
                    "method": "m",
                    "split": "construction",
                    "scene_id": "s",
                    "realization": 0,
                    "actor": actor,
                    "opportunity": opportunity,
                    "score": 3 + 2 * a - 4 * b + 7 * a * b,
                }
            )
    frame = pd.DataFrame(rows)
    terms = factorial_differences(frame).set_index("effect").delta
    assert terms.actor_x_opportunity == 7 and terms.actor_voluntary == 2
    assert terms.actor_assigned == 9
    with pytest.raises(ValueError, match="complete"):
        factorial_differences(frame.iloc[:-1])


def test_same_criterion_material_pairing(monkeypatch):
    import scripts.analyze_grain_state_opportunity as module

    monkeypatch.setattr(
        module, "paired_delta", lambda x, z, y: paired_delta(x, z, y, draws=12)
    )
    rng = np.random.default_rng(68)
    x, z = rng.normal(size=(2, 46))
    ids = [f"private_{i}" for i in range(46)]
    criteria = pd.DataFrame(
        {f"{t}_q": x + rng.normal(size=46) for t in ("baseline", "pre", "mid", "post")},
        index=ids,
    )
    frame = pd.DataFrame(
        [
            {
                "participant_code": uid,
                "reference": "R",
                "query": "q",
                "view": view,
                "score": vals[i],
                "native_score": vals[i] + 0.01 * i,
            }
            for view, vals in (("free_chat", x), ("direct_answers", z))
            for i, uid in enumerate(ids)
        ]
    )
    rows, pairs, timing, changes = cross_occasion_links(
        frame.sample(frac=1, random_state=9), criteria
    )
    assert (
        len(rows) == 16 and len(pairs) == 8 and len(timing) == 4 and len(changes) == 2
    )
    row = pairs.query("criterion=='post_q' and readout=='score'").iloc[0]
    assert row.delta_r_free_minus_direct == pytest.approx(
        np.corrcoef(x, criteria.post_q)[0, 1] - np.corrcoef(z, criteria.post_q)[0, 1]
    )

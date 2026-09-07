"""Synthetic projection/material tests; no model or private data access."""

import numpy as np
import pytest

from scripts.run_grain_attribute_direction import components, make_directions, material


def test_projection_and_complement():
    """Projected and complementary displacements add exactly; null norm matches."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 6.0, 8.0])
    d, q = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    p = components(x, y, d, q)
    np.testing.assert_allclose(p["projected"] + p["complement"] - x, y)
    assert np.dot(p["complement"] - x, d) == 0
    assert np.linalg.norm(p["matched_random"] - x) == np.linalg.norm(p["projected"] - x)


def test_frozen_direction_and_rotation():
    """Training alone sets fixed unit directions; projection is rotation equivariant."""
    rng = np.random.default_rng(14)
    c = rng.normal(size=(12, 4, 6))
    order = [{"entity": e} for e in ("Alex", "Blair", "room") for _ in range(4)]
    a = make_directions(c, order)
    rotation = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    b = make_directions(c @ rotation, order)
    for e in a:
        assert np.linalg.norm(a[e]["direction"]) == pytest.approx(1)
        assert np.linalg.norm(a[e]["random"]) == pytest.approx(1)
        assert abs(np.dot(a[e]["direction"], a[e]["random"])) < 1e-12
        np.testing.assert_allclose(
            a[e]["direction"] @ rotation, b[e]["direction"], atol=1e-12
        )


def test_incomplete_direction_training_rejected():
    """Never silently fit a smaller favorable training subset."""
    with pytest.raises(ValueError, match="Four training"):
        make_directions(np.zeros((3, 4, 5)), [{"entity": "Alex"}] * 3)


def test_held_material_ownership_and_prefix():
    """All crossed prompts keep role truth, target and within-person prefix aligned."""
    parent = []
    for entity in ("Alex", "Blair", "room"):
        for criterion in ("venue_object",) if entity == "room" else ("event", "trait"):
            prefix = "Example.\n\nFor "
            parent.append(
                {
                    "entity": entity,
                    "criterion": criterion,
                    "evidence": "Example.",
                    "user": prefix + entity + ", " + criterion,
                    "anchor_span": [len(prefix), len(prefix) + len(entity)],
                }
            )
    rows = material(parent)
    assert len(rows) == 80
    prefixes = {}
    for r in rows:
        a, b = r["anchor_span"]
        assert r["user"][a:b] == r["entity"]
        key = r["group_id"], r["level"], r["room_sign"], r["entity"]
        assert prefixes.setdefault(key, r["user"][:b]) == r["user"][:b]
        if r["entity"] == "room":
            assert r["expected_sign"] == r["room_sign"]
        else:
            line = next(
                x for x in r["evidence"].splitlines() if x.startswith(r["entity"] + " ")
            )
            initiated = "walked up" in line or "approached" in line
            assert initiated == (r["expected_sign"] == 1)
    assert len(prefixes) == 48

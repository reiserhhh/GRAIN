"""Material semantics and role separation contracts; not external validation."""

import pytest

from grain_research.gptpro_foundation_v3.f08_materials import (
    material_bank,
    render_event,
)


def test_bank_complete_unique_and_disjoint():
    """Each role has its own wrapper family and all status/pole/variant cells."""
    rows = material_bank()
    assert len(rows) == len({r["text"] for r in rows}) == 96
    for role in ("source", "validation", "application"):
        subset = [r for r in rows if r["role"] == role]
        assert len(subset) == 32
        assert len({r["family"] for r in subset}) == 1
        assert sum(r["qualification"] for r in subset) == 16
    assert all(r["material_id"] not in r["text"] for r in rows)


def test_correction_has_actual_final_pole():
    """Correction denies the opposite claim and explicitly states the actual action."""
    for pole, action, opposite in (
        (0, "rejected", "supported"),
        (1, "supported", "rejected"),
    ):
        text = render_event(pole, "negated", "plain", 0)
        assert f"I {opposite} the plan is incorrect" in text
        assert text.endswith(f"I actually {action} the plan.")


def test_invalid_material_rejected():
    """No silent wrapping of an out-of-bank variant or nonbinary label."""
    with pytest.raises(ValueError):
        render_event(2, "actual", "plain", 0)
    with pytest.raises(ValueError):
        render_event(0, "actual", "plain", 4)

"""Synthetic paired-order and comparison-denominator checks."""

import pandas as pd
import pytest

from scripts.run_grain_direction_order import compare_orders, reverse_material


def test_reversal_preserves_every_content_and_question():
    """Exactly two lines reverse without altering their lexical contents."""
    old = []
    for i in range(64):
        evidence = f"Header {i}.\nAlex started chatting.\nBlair waited."
        old.append(
            {
                "case_id": str(i),
                "entity": "Alex",
                "evidence": evidence,
                "user": evidence + "\nQuestion?",
            }
        )
    new = reverse_material(old)
    for a, b in zip(old, new):
        assert b["evidence"].splitlines() == [
            a["evidence"].splitlines()[j] for j in (0, 2, 1)
        ]
        assert b["user"].endswith("\nQuestion?")
        assert b["parent_case"] == a["case_id"]
    again = reverse_material(new)
    assert [r["evidence"] for r in again] == [r["evidence"] for r in old]


def test_incomplete_material_is_rejected():
    """Never silently use a small convenient subset."""
    with pytest.raises(ValueError, match="64"):
        reverse_material([])


def test_paired_summary_identity_and_missing_row():
    """All thirty setting/pooled cells must match with a one-to-one merge."""
    rows = []
    metrics = (
        "natural_rms",
        "shift_rms",
        "error_rms",
        "relative_rms",
        "relative_error",
        "intended_fraction",
    )
    for group in (None, "a", "b", "c", "d"):
        for criterion in ("event", "trait"):
            for patch in ("natural", "projected", "complement"):
                rows.append(
                    {
                        "scope": "pooled" if group is None else "setting",
                        "group_id": group,
                        "criterion": criterion,
                        "axis": "B",
                        "patch": patch,
                        **dict.fromkeys(metrics, 0.25),
                    }
                )
    frame = pd.DataFrame(rows)
    result = compare_orders(frame, frame)
    assert len(result) == 30
    assert (result[[k + "_reversed_minus_original" for k in metrics]] == 0).all().all()
    with pytest.raises(ValueError, match="population"):
        compare_orders(frame.iloc[:-1], frame)

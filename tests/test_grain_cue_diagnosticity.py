"""Synthetic full-factorial cue and semantic-code tests."""

import pandas as pd

from scripts.run_grain_attribute_direction import material as original_material
from scripts.run_grain_cue_diagnosticity import material, summaries


def templates():
    """Minimal synthetic parent questions independent of model/tokenizer files."""
    rows = []
    for entity in ("Alex", "Blair", "room"):
        for criterion in ("venue_object",) if entity == "room" else ("event", "trait"):
            prefix = "Example.\n\nFor "
            rows.append(
                {
                    "entity": entity,
                    "criterion": criterion,
                    "evidence": "Example.",
                    "user": prefix
                    + entity
                    + ", "
                    + criterion
                    + "? Return A for Yes and B for No.",
                    "anchor_span": [len(prefix), len(prefix) + len(entity)],
                }
            )
    return rows


def test_material_preserves_positive_actions_and_target():
    """All256 prompts preserve the initiating actor and exact queried entity."""
    original = original_material(templates())
    old = {r["case_id"]: r for r in original}
    rows = material(original)
    assert len(rows) == 256
    for r in rows:
        p = old[r["parent_case"]]
        positive = next(
            x
            for x in p["evidence"].splitlines()
            if "walked up" in x or "approached" in x
        )
        assert positive in r["evidence"].splitlines()
        a, b = r["anchor_span"]
        assert r["user"][a:b].strip() == r["entity"]
        if r["contrast"] == "avoidance":
            assert "declined" in r["evidence"] or "refused" in r["evidence"]
        assert ("Return A for Yes" in r["user"]) == (r["code_sign"] == 1)


def test_semantic_code_is_not_fixed_a_preference():
    """A perfectly code-equivariant synthetic score has all positive role gaps."""
    rows = material(original_material(templates()))
    frame = pd.DataFrame(rows)
    frame["margin"] = frame.role_sign * frame.code_sign
    _, gaps, summary, pairs = summaries(frame)
    assert len(gaps) == 128 and len(summary) == 80 and len(pairs) == 128
    assert (gaps.gap == 2).all()
    assert (summary.authored_contrast_preference == 1).all()
    assert pairs.semantic_preference_agrees.all()

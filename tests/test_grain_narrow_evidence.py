#!/usr/bin/env python3
"""Behavioral packet identities and affine responses need no model or private data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.narrow_evidence import (
    build_worlds,
    paired_response_tables,
    reader_links,
    response_quality,
)

ROOT = Path(__file__).resolve().parents[1]


def specification():
    """Public authored descriptions only; no historical participant fixtures."""
    return json.loads((ROOT / "configs/grain_narrow_evidence_v1.json").read_text())


def definitions():
    """Derive public family names without loading private question artifacts."""
    spec = specification()
    names = {
        "anthro_total": "ai_mind_total",
        "anthro_agency": "ai_mind_agency",
        "anthro_experience": "ai_mind_experience",
        "attachment_anxiety": "ai_attachment_anxiety",
        "attachment_avoidance": "ai_attachment_avoidance",
        "trust_positive": "ai_trust_positive",
        "distrust": "ai_distrust",
        "leak_concern": "ai_privacy_concern",
        "perceived_empathy": "perceived_ai_empathy",
        "disclosability_mean_exploratory": "disclosure_composite",
        **{f"disclosability_lv{k}": f"disclosure_depth_{k}" for k in range(1, 5)},
    }
    result = {}
    for q in spec["related"]:
        short = q.removeprefix("pre_").removeprefix("post_")
        result[q] = {"family": names.get(short, short)}
    return result


def numerical_scores():
    """Build a known affine instrument with nonconstant native action amplitude."""
    rng = np.random.default_rng(71)
    ref = FunctionalQuery(
        rng.normal(size=(4, 3)),
        rng.normal(size=3),
        rng.normal(size=(4, 3)),
        rng.normal(size=3),
        0.4,
    )
    rows = []
    for pair in range(2):
        for owner in ("self", "other"):
            for polarity in ("high", "low"):
                parts = ref.parts(rng.normal(size=(4, 3)), rng.normal(size=3))
                base = {
                    "target": "q",
                    "query": "q",
                    "reader_role": "target",
                    "pair": pair,
                    "owner": owner,
                    "polarity": polarity,
                }
                rows.append(base | {"readout": "affine", **parts})
                rows.append(
                    base
                    | {
                        "readout": "native",
                        "score": parts["score"] * 2,
                        "mean": np.nan,
                        "configuration": np.nan,
                        "final": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def test_public_world_coverage_and_attribution():
    worlds = build_worlds(definitions(), specification())
    assert len(worlds) == len({w["world_id"] for w in worlds}) == 200
    assert len({w["family"] for w in worlds}) == 20
    for i in range(0, len(worlds), 2):
        own, other = worlds[i : i + 2]
        assert own["owner"] == "self" and other["owner"] == "other"
        assert own["evidence_body"] == other["evidence_body"]
        assert own["texts"] != other["texts"]
        assert own["texts"][0].split('"')[1] == other["texts"][0].split('"')[1]


def test_reader_coverage():
    links = reader_links(definitions(), specification())
    assert len(links) == 75
    for target in definitions():
        group = [r for r in links if r["target"] == target]
        assert len({r["query"] for r in group}) == 3
        assert {r["reader_role"] for r in group} == {"target", "related", "distant"}


@pytest.mark.parametrize("mutation", ("same", "empty", "missing"))
def test_invalid_evidence_rejected(mutation):
    spec = specification()
    if mutation == "same":
        spec["pairs"]["ai_mind_total"][0] = ["text", "text"]
    elif mutation == "empty":
        spec["pairs"]["ai_mind_total"][0][0] = ""
    else:
        del spec["pairs"]["ai_mind_total"]
    with pytest.raises(ValueError):
        build_worlds(definitions(), spec)


def test_duplicate_reader_rejected():
    spec = specification()
    spec["related"]["pre_anthro_agency"] = "pre_anthro_agency"
    with pytest.raises(ValueError):
        reader_links(definitions(), spec)


def test_polarity_and_attribution_arithmetic():
    paired, attribution = paired_response_tables(numerical_scores())
    assert len(paired) == 4 and len(attribution) == 2
    for frame in (paired, attribution):
        np.testing.assert_allclose(frame.native, 2 * frame.affine)
        np.testing.assert_allclose(
            frame.affine, frame[["mean", "configuration", "final"]].sum(axis=1)
        )
    q = response_quality(paired.native, paired.affine)
    assert q["response_r"] == pytest.approx(1)
    assert q["amplitude_slope"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_polarity",
        "missing_owner",
        "duplicate",
        "component",
        "nonfinite",
        "missing_reader",
    ),
)
def test_incomplete_pairs_rejected(mutation):
    scores = numerical_scores()
    if mutation == "missing_polarity":
        scores = scores.iloc[1:]
    elif mutation == "missing_owner":
        scores = scores[scores.owner == "self"]
    elif mutation == "duplicate":
        scores = pd.concat([scores, scores.iloc[[0]]])
    elif mutation == "component":
        scores.loc[0, "mean"] += 1
    elif mutation == "nonfinite":
        scores.loc[0, "score"] = np.nan
    else:
        scores = scores[scores.readout == "native"]
    with pytest.raises(ValueError):
        paired_response_tables(scores)


def test_zero_native_effect_is_not_perfect_fidelity():
    quality = response_quality(np.zeros(4), np.ones(4))
    assert quality["response_r"] is None
    assert quality["amplitude_slope"] is None
    assert quality["sign_agreement_over_resolution"] is None
    assert quality["mae"] == 1


def test_order_independent_response_pairing():
    scores = numerical_scores()
    expected = paired_response_tables(scores)
    actual = paired_response_tables(scores.sample(frac=1, random_state=11))
    for a, b in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(a, b)

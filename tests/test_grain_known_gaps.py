#!/usr/bin/env python3
"""Synthetic checks for upstream coordinate comparison and feedback integrity."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.natural_anchors import (
    expand_bank,
    fingerprint,
    promote_feedback,
    retrieve,
)
from scripts.fetch_grain_known_gap_resources import read_registry
from scripts.run_grain_known_native import compare_coordinates
from scripts.run_grain_text_native import aligned_oof

ROOT = Path(__file__).resolve().parents[1]


def test_bank_scopes_keys_and_provenance():
    bank = expand_bank(
        json.loads((ROOT / "configs/grain_natural_anchors_v1.json").read_text())
    )
    assert len(bank) == 60
    assert sum(r["scope"] == "episode" for r in bank) == 20
    assert all(not r["expert_validated"] for r in bank)
    assert {r["key"] for r in bank} == {-1, 1}
    assert len({r["facet"] for r in bank}) == 10


def test_retrieve_top_one_and_threshold():
    ix, score, accepted = retrieve(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), np.eye(2), [{}, {}]
    )
    assert ix.tolist() == [0, 1, 1]
    assert score.tolist() == [1.0, 1.0, 0.0]
    assert accepted.tolist() == [True, True, False]
    with pytest.raises(ValueError, match="Zero"):
        retrieve(np.zeros((1, 2)), np.eye(2), [{}, {}])


def proposal_and_reviews(kind="model"):
    p = {
        "id": "new",
        "text": "I tend to help people.",
        "facet": "A3",
        "key": 1,
        "scope": "habitual",
        "source_partition": "development",
    }
    reviews = [
        {
            "proposal_id": "new",
            "proposal_sha256": fingerprint(p),
            "rater_id": str(i),
            "rater_kind": kind,
            "supports": True,
        }
        for i in range(2)
    ]
    return p, reviews


def test_feedback_never_masquerades_as_expert():
    p, ratings = proposal_and_reviews()
    assert promote_feedback([], [p], ratings)[0] == []
    bank, decisions = promote_feedback([], [p], ratings, require_human=False)
    assert decisions[0]["promoted"]
    assert not bank[0]["expert_validated"] and not bank[0]["human_reviewed"]
    assert bank[0]["trait"] == "Agreeableness"


@pytest.mark.parametrize(
    "case", ["hash", "duplicate", "heldout", "criterion", "verdict"]
)
def test_feedback_rejects_contaminated_inputs(case):
    p, ratings = proposal_and_reviews("human")
    if case == "hash":
        p["text"] += " changed"
    elif case == "duplicate":
        ratings[1]["rater_id"] = ratings[0]["rater_id"]
    elif case == "heldout":
        p["source_partition"] = "validation"
        for r in ratings:
            r["proposal_sha256"] = fingerprint(p)
    elif case == "criterion":
        p["criterion_selected"] = True
        for r in ratings:
            r["proposal_sha256"] = fingerprint(p)
    else:
        ratings[0]["supports"] = "yes"
    with pytest.raises(ValueError):
        promote_feedback([], [p], ratings)


def test_negative_feedback_blocks_promotion():
    p, ratings = proposal_and_reviews("human")
    ratings[1]["supports"] = False
    assert promote_feedback([], [p], ratings)[0] == []


def test_coordinate_comparison_ignores_axis_sign_and_permutation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 4))
    y = x[:, [2, 0, 3, 1]] * [-1, 1, -1, 1]
    result = compare_coordinates(x, y)
    assert result["minimum_matched_absolute_r"] > 0.999999
    assert result["relative_procrustes_error"] < 1e-12


def test_registry_with_multiline_preamble(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text(
        '"preamble\nmore text",x\nName,Path\nmodel,https://example.org/model\n,\n'
    )
    table = read_registry(path)
    assert table.Name.tolist() == ["model"]


def test_prior_oof_identity_alignment(tmp_path):
    p = tmp_path / "oof.npz"
    ids = np.array(["u1", "u2"])
    folds = np.array([0, 1])
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.savez(p, author_ids=ids, folds=folds, y=y, semantic_pca_ridge=y)
    assert np.array_equal(aligned_oof(p, ids, folds, y), y)
    with pytest.raises(ValueError, match="cohort/order"):
        aligned_oof(p, ids[::-1], folds, y)

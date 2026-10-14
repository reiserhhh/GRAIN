#!/usr/bin/env python3
"""Focused tests of typed binding, holdout separation and parser lineage."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_binding as B


def test_material_roles_isolate_all_variants_and_unseen_objects():
    spec = json.loads(
        (Path(__file__).parents[1] / "configs/grain_math_binding_v1.json").read_text()
    )
    bank = B.material_bank(spec)
    assert len(bank) == 1536
    groups = {}
    for r in bank:
        groups.setdefault(r["pair_id"], set()).add(r["role"])
        assert (r["object_index"] >= 8) == (r["role"] == "unseen_object")
    assert all(len(v) == 1 for v in groups.values())
    controls = B.relation_controls(spec, bank)
    for a, b in zip(controls[::2], controls[1::2], strict=True):
        assert sorted(a["text"].split()) == sorted(b["text"].split())
        assert a["expected_present"] and not b["expected_present"]


def test_product_kernel_matches_explicit_typed_tensor():
    rng = np.random.default_rng(61)
    a, b = rng.normal(size=(7, 3)), rng.normal(size=(7, 4))
    aa, bb = np.c_[np.ones(7), a], np.c_[np.ones(7), b]
    phi = np.einsum("ni,nj->nij", aa, bb).reshape(7, -1)
    np.testing.assert_allclose(
        B.pair_kernel(a, b, a, b, "tensor"), phi @ phi.T, atol=1e-12
    )


def test_kernel_cold_scores_match_primal_centered_ridge():
    rng = np.random.default_rng(52)
    a, b = rng.normal(size=(8, 2)), rng.normal(size=(8, 2))
    y = a[:, 0] * b[:, 0]
    model = B.fit_kernel(a, b, y, alpha=0.1)
    phi = np.einsum("ni,nj->nij", np.c_[np.ones(8), a], np.c_[np.ones(8), b]).reshape(
        8, -1
    )
    centered = phi - phi.mean(axis=0)
    w = np.linalg.solve(
        centered.T @ centered + 0.1 * np.eye(phi.shape[1]), centered.T @ (y - y.mean())
    )
    np.testing.assert_allclose(
        B.score_kernel(model, a, b), centered @ w + y.mean(), atol=1e-11
    )


def test_selection_never_reads_test_vectors_or_labels():
    rng = np.random.default_rng(52)
    a, b = rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    y = rng.normal(size=12)
    roles = np.array(["source"] * 6 + ["validation"] * 3 + ["unseen_pair"] * 3)
    first, meta = B.select_kernel(a, b, y, roles, "tensor")
    a[9:], b[9:], y[9:] = 1e7, -1e8, 1e12
    second, meta2 = B.select_kernel(a, b, y, roles, "tensor")
    assert meta == meta2
    for k in first:
        np.testing.assert_array_equal(first[k], second[k])


def test_and_has_real_interaction_and_additive_residual():
    witness = B.exact_binding_witness()
    assert witness["additive_rmse"] == pytest.approx(0.5)
    assert witness["tensor_rmse"] < 1e-12
    assert witness["interaction_contrast"] == 2


def test_role_binding_not_bag_of_words_for_conjoined_predicates():
    text = "I organize workers and photograph badges."
    words = []
    for i, (token, lemma, pos, head, rel) in enumerate(
        [
            ("I", "I", "PRON", 2, "nsubj"),
            ("organize", "organize", "VERB", 0, "root"),
            ("workers", "worker", "NOUN", 2, "obj"),
            ("and", "and", "CCONJ", 5, "cc"),
            ("photograph", "photograph", "VERB", 2, "conj"),
            ("badges", "badge", "NOUN", 5, "obj"),
        ]
    ):
        start = text.index(token)
        words.append(
            {
                "id": i + 1,
                "text": token,
                "lemma": lemma,
                "upos": pos,
                "head": head,
                "deprel": rel,
                "start": start,
                "end": start + len(token),
            }
        )
    events = B.dependency_slots([words], text)
    assert [(r["action"], r["object"]) for r in events] == [
        ("organize", "workers"),
        ("photograph", "badges"),
    ]


def test_unrelated_is_not_negative_polarity():
    z = np.array([[-1, 1, 1, 1], [1, -1, 1, 1]])
    scores, _ = B.product_scores(z)
    np.testing.assert_array_equal(scores, [0, -1])


def test_unknown_kernel_fails_explicitly():
    with pytest.raises(ValueError, match="unknown"):
        B.pair_kernel(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            "other",
        )


def test_cold_score_does_not_depend_on_query_batch():
    rng = np.random.default_rng(75)
    a, b = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    model = B.fit_kernel(a[:6], b[:6], a[:6, 0] * b[:6, 0])
    together = B.score_kernel(model, a[6:], b[6:])
    separate = [B.score_kernel(model, a[i : i + 1], b[i : i + 1])[0] for i in (6, 7)]
    np.testing.assert_allclose(together, separate, atol=1e-12)

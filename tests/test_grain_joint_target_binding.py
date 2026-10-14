#!/usr/bin/env python3
"""Synthetic controls for joint target binding, not a personality benchmark."""

import numpy as np
import pytest

from grain_research.construction import math_joint_target_binding as M


def test_complete_fixed_word_bags_and_addresses():
    rows = M.documents()
    assert len(rows) == 16
    rows[0] = {**rows[0], "text": rows[0]["text"] + " added"}
    with pytest.raises(ValueError, match="word bag"):
        M.validate_materials(rows)


def test_logical_joint_component_not_a_global_bag():
    rows = M.documents()
    y = M.truth_table(rows)
    fields = M.binding_fields(y)
    en = np.array([r["e"] * r["n"] for r in rows])
    np.testing.assert_allclose(fields["AND"], en / 4)
    np.testing.assert_allclose(fields["OR"], -en / 4)
    np.testing.assert_allclose(fields["logical_common"], 0)
    np.testing.assert_allclose(fields["composition_remainder"], 0)


def test_author_marginals_identical_joint_readings_different():
    rows = M.documents()
    y = M.truth_table(rows)
    difference = M.author_contrast(rows, y, 0.4)
    np.testing.assert_allclose(difference[:, :, :4], 0)
    np.testing.assert_allclose(
        difference[:, :, 4:8], [[[0.2, -0.2, -0.2, 0.2]] * 2] * 2
    )
    np.testing.assert_allclose(difference[:, :, 12:], 0)


def test_exact_character_reconstruction_and_query_roles():
    rows = M.documents()
    rng = np.random.default_rng(198)
    y = rng.normal(size=(16, 7))
    c = M.coefficients(rows, y)
    for i, row in enumerate(rows):
        sign = np.array([1, row["e"], row["n"], row["e"] * row["n"]])
        np.testing.assert_allclose(
            sign @ c[row["bank"], row["order"]], y[i], atol=1e-15
        )
    q = M.queries()
    assert len(q) == 14
    assert [r["kind"] for r in q[4:12]] == ["AND"] * 4 + ["OR"] * 4
    assert q[4]["e_actor"] == q[4]["n_actor"] == "A"
    assert q[5]["e_actor"] == "A" and q[5]["n_actor"] == "B"


def test_atomic_product_is_only_a_declared_composer():
    rows = M.documents()
    y = M.truth_table(rows)
    y[:, 4:8] = 0.5
    y[:, 8:12] = 0.5
    f = M.binding_fields(y)
    assert np.linalg.norm(f["composition_remainder"]) > 0
    np.testing.assert_allclose(f["logical_opposition"], 0)
    with pytest.raises(ValueError):
        M.binding_fields(np.zeros((16, 13)))


def test_cached_native_tokenizer_returns_explicit_token_list():
    """Catch template API drift without loading weights or observing answers."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if not MODEL.exists():
        pytest.skip("Local frozen tokenizer unavailable; no download")
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    row, query = M.documents()[0], M.queries()[0]
    messages = [
        {"role": "system", "content": M.SYSTEM},
        {
            "role": "user",
            "content": f"Record:\n{row['text']}\n\nQuestion: {query['question']}\nAnswer:",
        },
    ]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    assert isinstance(ids, list)
    assert ids == tok(rendered, add_special_tokens=False)["input_ids"]
    assert rendered.count(row["text"]) == 1

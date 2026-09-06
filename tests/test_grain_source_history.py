#!/usr/bin/env python3
"""History donor boundaries, exact current placement and masked-key instrumentation."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.source_history import (
    aligned_triplet,
    capture_aligned,
    profile,
    select_donor,
    split_view,
    tokenize_parts,
)
from grain_research.vector_mechanisms.trace import DecoderObserver


def comments(uid, offset=0):
    """Generate twelve unique timestamped comments, independent of real text."""
    return [
        {
            "author_id": uid,
            "comment_id": f"{uid}:{i}",
            "thread_id": f"thread:{uid}:{i}",
            "body": f"synthetic {uid} comment {i}",
            "timestamp": float(i + offset),
            "subreddit": "common",
        }
        for i in range(12)
    ]


def test_whole_view_support_rejects_silent_replacement():
    view = comments("a")
    h, c = split_view(view[::-1])
    assert len(h) == 11 and c == view[-1]
    for field, value in (
        ("timestamp", view[-2]["timestamp"]),
        ("thread_id", view[-2]["thread_id"]),
        ("body", view[-2]["body"]),
    ):
        bad = copy.deepcopy(view)
        bad[-1][field] = value
        with pytest.raises(ValueError):
            split_view(bad)
    with pytest.raises(ValueError):
        split_view(view[:-1])


def test_reference_history_selection_excludes_shared_material_and_future():
    target = comments("target", 100)
    h, current = split_view(target)
    own = profile(h, lambda s: len(s.split()))
    donors = []
    for uid, offset in (("d1", 0), ("d2", 20), ("future", 200)):
        hist, _ = split_view(comments(uid, offset))
        donors.append(
            {
                "author_id": uid,
                "view": 0,
                "history": hist,
                "profile": profile(hist, lambda s: len(s.split())),
            }
        )
    selected = select_donor(own, current, target, donors)
    assert selected["donor"]["author_id"] == "d2" and selected["eligible_donors"] == 2
    donors[1]["profile"]["threads"].add(target[0]["thread_id"])
    assert select_donor(own, current, target, donors)["donor"]["author_id"] == "d1"
    donors[0]["profile"]["bodies"].add(own["bodies"].pop())
    with pytest.raises(ValueError, match="No eligible"):
        select_donor(own, current, target, donors)


class CharTokenizer:
    """Simple deterministic chat tokenizer for boundary checks, not a model tokenizer."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = (
            "SYSTEM\n"
            + messages[0]["content"]
            + "\nUSER\n"
            + messages[1]["content"]
            + "\nASSISTANT\n"
        )
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class BoundaryTokenizer(CharTokenizer):
    """Mimic punctuation/newline merges that cross the raw history boundary."""

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        i = 0
        while i < len(text):
            if text[i : i + 3] == ".\n\n":
                ids.append(1000)
                offsets.append((i, i + 3))
                i += 3
            else:
                ids.append(ord(text[i]))
                offsets.append((i, i + 1))
                i += 1
        return {"input_ids": ids, "offset_mapping": offsets}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = super().apply_chat_template(messages, False, add_generation_prompt)
        return self(text)["input_ids"] if tokenize else text


def test_separator_envelope_repairs_boundary_without_retokenizing():
    tok = BoundaryTokenizer()
    own, donor = comments("own")[:11], comments("donor")[:11]
    own[-1]["body"] += "."
    donor[-1]["body"] += "!"
    before = [tokenize_parts(tok, h, "fixed current", "behavior") for h in (own, donor)]
    with pytest.raises(ValueError, match="differ"):
        aligned_triplet(*before, 0, max_length=4096)
    after = [
        tokenize_parts(tok, h, "fixed current", "behavior", enclose_separator=True)
        for h in (own, donor)
    ]
    for a, b in zip(before, after):
        assert a["native_ids"] == b["native_ids"]
        assert [a["suffix"][i] for i in a["current_in_suffix"]] == [
            b["suffix"][i] for i in b["current_in_suffix"]
        ]
    rows = aligned_triplet(*after, 0, max_length=4096)
    assert len({tuple(r["current_positions"]) for r in rows.values()}) == 1
    assert len({r["current_token_sha256"] for r in rows.values()}) == 1
    assert after[0]["suffix"] == after[1]["suffix"]


def test_chat_boundaries_and_fixed_current_triplet():
    tok = CharTokenizer()
    own = tokenize_parts(tok, comments("a")[:11], "constant current words", "behavior")
    donor = tokenize_parts(
        tok, comments("longer_name")[:11], "constant current words", "behavior"
    )
    rows = aligned_triplet(own, donor, 0, max_length=4096)
    assert len({tuple(r["current_positions"]) for r in rows.values()}) == 1
    assert len({r["current_token_sha256"] for r in rows.values()}) == 1
    for row in rows.values():
        assert (
            "".join(chr(row["ids"][i]) for i in row["current_positions"])
            == "constant current words"
        )
    s, e = rows["invisible"]["slot"]
    assert not any(rows["invisible"]["mask"][s:e])
    assert sum(rows["matched"]["mask"]) == len(donor["native_ids"])
    assert sum(rows["own"]["mask"]) == len(own["native_ids"])
    bad = copy.deepcopy(donor)
    bad["suffix"][0] += 1
    with pytest.raises(ValueError, match="differ"):
        aligned_triplet(own, bad, 0, 4096)
    with pytest.raises(ValueError, match="budget"):
        aligned_triplet(own, donor, 0, 40)


def test_tiny_decoder_mask_is_invisible_and_native_replays():
    torch.manual_seed(731)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    ).eval()
    model.config._attn_implementation = "eager"
    obs = DecoderObserver(
        model, SimpleNamespace(pad_token_id=0), max_tokens=64, pad_to_length=48
    )
    own = {
        "prefix": [2, 3],
        "history": [4, 5, 6],
        "suffix": [7, 8, 9, 10],
        "current_in_suffix": [0, 1],
    }
    other = own | {"history": [11, 12, 13]}
    rows = aligned_triplet(own, other, 0, max_length=48)
    row = rows["own"]
    native, nlog = obs.capture(
        row["ids"],
        positions=[row["decision_position"]],
        pools=[("mean", row["current_positions"]), ("std", row["current_positions"])],
    )
    actual, alog = capture_aligned(obs, row)
    for k in native:
        np.testing.assert_array_equal(native[k], actual[k])
    np.testing.assert_array_equal(nlog, alog)
    invisible = rows["invisible"]
    before, blog = capture_aligned(obs, invisible)
    changed = copy.deepcopy(invisible)
    s, e = changed["slot"]
    changed["ids"][s:e] = [19] * (e - s)
    after, clog = capture_aligned(obs, changed)
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])
    np.testing.assert_array_equal(blog, clog)
    matched, _ = capture_aligned(obs, rows["matched"])
    np.testing.assert_array_equal(actual["input_embedding"], matched["input_embedding"])
    assert not np.array_equal(actual["block_01"], matched["block_01"])
    restored, rlog = capture_aligned(obs, row)
    np.testing.assert_array_equal(rlog, alog)
    for k in actual:
        np.testing.assert_array_equal(actual[k], restored[k])
    invalid = copy.deepcopy(row)
    invalid["current_positions"] = [-1]
    with pytest.raises(ValueError, match="Invalid"):
        capture_aligned(obs, invalid)


def test_full_layout_arm_task_completeness_and_reference_role():
    import pandas as pd

    from scripts.prepare_grain_source_history import check_layouts, reference_ids

    own, current = split_view(comments("own"))
    donor, _ = split_view(comments("donor"))
    rows = check_layouts(CharTokenizer(), own, donor, current)
    assert len(rows) == 18
    assert len({(r["text_arm"], r["task"], r["history_arm"]) for r in rows}) == 18
    assert all(r["prompt_tokens"] + 16 <= 3072 for r in rows)
    source = [{"author_id": str(i), "source_group": "mbti_only"} for i in range(100)]
    budget = pd.DataFrame(
        {
            "author_id": [str(i) for i in range(100)],
            "old_reference_member": [True] * 99 + [False],
        }
    )
    assert reference_ids(source, budget) == {str(i) for i in range(99)}
    bad = budget.copy()
    bad["old_reference_member"] = bad.old_reference_member.astype(str)
    with pytest.raises(ValueError, match="boolean"):
        reference_ids(source, bad)
    source[0]["source_group"] = "bridge"
    with pytest.raises(ValueError, match="MBTI-only"):
        reference_ids(source, budget)


def test_instrument_reconstructs_frozen_layouts_without_rematching():
    from grain_research.vector_mechanisms.history_instrument import material_layouts
    from scripts.prepare_grain_source_history import check_layouts

    tok = BoundaryTokenizer()
    own = {"author_id": "own", "source_group": "big5_only", "views": []}
    donor = {"author_id": "donor", "source_group": "mbti_only", "views": []}
    case = {"author_id": "own", "source_group": "big5_only", "views": []}
    for vi in range(3):
        view = comments(f"own:{vi}", 100)
        other = comments(f"donor:{vi}")
        own["views"].append(view)
        donor["views"].append(other)
        h, c = split_view(view)
        d, _ = split_view(other)
        case["views"].append(
            {
                "view": vi,
                "current_id": c["comment_id"],
                "own_history_ids": [x["comment_id"] for x in h],
                "donor_author_id": "donor",
                "donor_view": vi,
                "donor_history_ids": [x["comment_id"] for x in d],
                "layouts": check_layouts(tok, h, d, c, enclose_separator=True),
            }
        )
    source = {"own": own, "donor": donor}
    assert len(material_layouts(case, source, tok)) == 54
    bad = copy.deepcopy(case)
    bad["views"][0]["layouts"][0]["layout_sha256"] = "changed"
    with pytest.raises(ValueError, match="commitment"):
        material_layouts(bad, source, tok)


def test_full_history_triplet_instrument_with_interior_padding():
    from grain_research.vector_mechanisms.history_instrument import qualify_triplet

    torch.manual_seed(731)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    ).eval()
    model.config._attn_implementation = "eager"
    obs = DecoderObserver(
        model, SimpleNamespace(pad_token_id=0), max_tokens=64, pad_to_length=48
    )
    own = {
        "prefix": [2, 3],
        "history": [4, 5, 6],
        "suffix": [7, 8, 9, 10],
        "current_in_suffix": [0, 1],
    }
    matched = own | {"history": [11, 12, 13, 14, 15]}
    rows = aligned_triplet(own, matched, 0, max_length=48)
    captured, checks = qualify_triplet(obs, rows)
    assert checks["forwards"] == 6
    assert all(v == 0 for k, v in checks.items() if k.endswith("error"))
    assert not np.array_equal(
        captured["own"][0]["block_01"], captured["matched"][0]["block_01"]
    )


def test_instrument_archive_includes_entrypoints_not_secrets(tmp_path):
    import zipfile

    from grain_research.vector_mechanisms.history_instrument import archive_instrument

    for name in (
        "scripts/qualify_grain_source_history.py",
        "scripts/prepare_grain_source_history.py",
        "docs/GRAIN_VM05_HISTORY_COUNTERFACTUAL_V1.md",
        "tests/test_grain_source_history.py",
        ".env",
    ):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic fixture")
    output = tmp_path / "archive.zip"
    archive_instrument(tmp_path, output)
    with zipfile.ZipFile(output) as z:
        names = set(z.namelist())
        assert "scripts/qualify_grain_source_history.py" in names
        assert "scripts/prepare_grain_source_history.py" in names
        assert "tests/test_grain_source_history.py" in names
        assert ".env" not in names


def test_real_history_observer_rejects_live_queue_before_loading(tmp_path):
    import json

    from scripts.qualify_grain_source_history import load_observer

    (tmp_path / "queue_status.json").write_text(json.dumps({"status": "RUNNING"}))
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="must finish"):
        load_observer(SimpleNamespace(observer_queue_dir=tmp_path), {})

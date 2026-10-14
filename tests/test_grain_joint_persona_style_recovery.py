#!/usr/bin/env python3
"""Exercise the actual local tokenizer contract without loading model weights."""

import pytest

from scripts.prepare_grain_joint_persona_style_hm150_recovery import flat_ids
from scripts.run_grain_joint_persona_style_hm150 import H, M, MODEL


def test_mapping_and_list_supported():
    assert flat_ids({"input_ids": [1, 2], "attention_mask": [1, 1]}) == [1, 2]
    assert flat_ids([1, 2]) == [1, 2]
    with pytest.raises(ValueError, match="integer"):
        flat_ids(["input_ids", "attention_mask"])


def test_actual_cached_template_contract():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows = M.joint_materials(H.load(H.SOURCE / "manifest.json"))
    for row in rows:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
        ]
        ids = flat_ids(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        raw = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert ids == tokenizer.encode(raw, add_special_tokens=False)
        assert len(ids) < 384

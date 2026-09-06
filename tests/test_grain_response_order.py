"""Synthetic invariant tests for exact-token-bag context-order interventions."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.response_order import (
    MODES,
    SEEDS,
    assistant_span,
    permute_prefix,
    summarize_order,
)


class CharacterTokenizer:
    """Character offsets expose role boundaries without any private text or weights."""

    all_special_ids = ()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not add_generation_prompt
        rendered = "".join(f"[{m['role']}]\n{m['content']}\n" for m in messages)
        return {"input_ids": self.encode(rendered)} if tokenize else rendered

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_native_role_and_response_boundaries_stay_fixed():
    context, response = (
        "A long preceding assistant context.",
        "The current response stays fixed.",
    )
    prefix, body, indices, scope = assistant_span(
        CharacterTokenizer(), response, context
    )
    assert [prefix[i] for i in indices] == [ord(c) for c in context]
    assert body == [ord(c) for c in response]
    assert scope["fixed_boundary_crossing_tokens"] == 0
    for mode in MODES:
        for seed in SEEDS:
            changed, record = permute_prefix(prefix, indices, "C001", mode, seed)
            assert sorted(changed) == sorted(prefix)
            assert changed[: indices[0]] == prefix[: indices[0]]
            assert changed[indices[-1] + 1 :] == prefix[indices[-1] + 1 :]
            assert record["changed_tokens"] > 0
            repeat, same = permute_prefix(prefix, indices, "C001", mode, seed)
            assert repeat == changed and same == record


def test_block_order_keeps_internal_eight_token_sequences():
    changed, _ = permute_prefix(list(range(41)), list(range(1, 34)), "X", "block8", 731)
    isolated = changed[1:34]
    for start in range(1, 34, 8):
        group = list(range(start, min(start + 8, 34)))
        pos = isolated.index(group[0])
        assert isolated[pos : pos + len(group)] == group


def test_donor_order_does_not_depend_on_recipient_suffix():
    context = list(range(20, 52))
    a, am = permute_prefix(
        [1, 2] + context + [3], list(range(2, 34)), "donor", "token", 1731
    )
    b, bm = permute_prefix(
        [1, 2] + context + [9, 9], list(range(2, 34)), "donor", "token", 1731
    )
    assert a[2:34] == b[2:34]
    assert am == bm


def test_degenerate_repeated_tokens_are_reported_not_resampled():
    changed, meta = permute_prefix([3] * 24, list(range(24)), "same", "token", 731)
    assert changed == [3] * 24 and meta["changed_tokens"] == 0
    one, single = permute_prefix(list(range(8)), list(range(8)), "small", "block8", 731)
    assert one == list(range(8)) and single["permutation_groups"] == 1


def test_invalid_or_special_context_is_not_silently_used():
    with pytest.raises(ValueError):
        permute_prefix(list(range(24)), list(range(24)), "x", "best_seed_search", 731)
    t = CharacterTokenizer()
    t.all_special_ids = [ord("z")]
    with pytest.raises(ValueError, match="special token"):
        assistant_span(t, "response text", "a long z-containing context")


def test_common_corruption_penalty_cancels_in_interaction():
    rows = []
    for i in range(4):
        for role, base in (("actual", -3.0), ("matched", -3.4)):
            rows.append(
                {
                    "participant_code": str(i),
                    "condition": role + "_native",
                    "mean_logp": base,
                    "tail_logp": base,
                }
            )
            for mode in MODES:
                for seed in SEEDS:
                    rows.append(
                        {
                            "participant_code": str(i),
                            "condition": f"{role}_{mode}_{seed}",
                            "mean_logp": base - 0.5,
                            "tail_logp": base - 0.5,
                        }
                    )
    effects = summarize_order(pd.DataFrame(rows))
    for mode in MODES:
        out = effects[f"mean_logp/{mode}/seed_mean"]
        assert abs(out["order_interaction"]["mean"]) < 1e-12
        assert np.isclose(out["common_degradation"]["mean"], 0.5)
        assert out["order_interaction"]["participants"] == 4

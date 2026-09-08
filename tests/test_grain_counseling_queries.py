#!/usr/bin/env python3
"""Query object separation and label-independent evidence coverage."""

from itertools import pairwise

from grain_research.vector_mechanisms.counseling_queries import (
    questions,
    render,
    select_evidence,
)


class CharacterTokenizer:
    """One synthetic token per character; no dependency on a downloaded model."""

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_all_scored_occasions_have_separate_referents():
    q = questions()
    assert len(q) == 22
    for family in (
        "anthro_total",
        "anthro_agency",
        "anthro_experience",
        "attachment_anxiety",
        "attachment_avoidance",
    ):
        assert q[f"pre_{family}"]["question"] != q[f"post_{family}"]["question"]
    assert "general" in q["pre_anthro_total"]["question"]
    assert "involved in this interaction" in q["post_anthro_total"]["question"]


def test_complete_small_material_and_even_long_coverage():
    tok = CharacterTokenizer()
    texts = ["abc", "日本語の発言"]
    selected, audit = select_evidence(texts, tok, budget=100)
    assert selected == texts and all(a["complete"] for a in audit)
    text = "abcdefghij" * 20
    selected, audit = select_evidence([text], tok, budget=40)
    spans = audit[0]["char_spans"]
    assert len(selected) == 4 and sum(map(len, selected)) == 40
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert selected == [text[a:b] for a, b in spans]
    assert all(a[1] <= b[0] for a, b in pairwise(spans))


def test_prompt_keeps_scenario_and_ai_authorship_separate():
    prompt = render(["participant text"], "post_perceived_empathy", "direct_answers")
    assert "Scenario events need not have happened" in prompt
    assert "not the AI's actual abilities" in prompt
    assert prompt.count("Question:") == 1

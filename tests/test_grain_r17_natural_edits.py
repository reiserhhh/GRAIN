"""Edit-function contracts for the R17 natural sensitivity packet."""

import collections
import re

from scripts.run_grain_r17_natural_sensitivity import edit_text

TEXT = "I don't think so. You're WRONG! It's fine, isn’t it? DON'T go."


def test_case_fold_retains_casefolded_tokens():
    new, app, ret, _ = edit_text(TEXT, "case_fold")
    assert app and ret and new == TEXT.lower()


def test_contraction_expand_keeps_outside_text_and_case():
    new, app, ret, det = edit_text(TEXT, "contraction_expand")
    assert app and ret and det["substitutions"] == 4
    assert "You are" in new and "DO NOT" in new and "It's" in new


def test_sentence_reverse_preserves_token_multiset():
    new, app, ret, det = edit_text(TEXT, "sentence_reverse")
    assert app and ret and det["segments"] == 4
    assert collections.Counter(new.split()) == collections.Counter(TEXT.split())
    assert new.startswith("DON'T go.")


def test_sentence_reverse_refuses_short_segment():
    assert not edit_text("Hello there. Ok.", "sentence_reverse")[1]


def test_punctuation_strip_keeps_word_sequence():
    new, app, ret, _ = edit_text(TEXT, "punctuation_strip")
    assert app and ret
    assert re.findall(r"\w+", new) == re.findall(r"\w+", TEXT)
    assert "!" not in new and "?" not in new and "don't" in new

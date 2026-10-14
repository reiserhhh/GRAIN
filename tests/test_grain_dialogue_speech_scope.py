#!/usr/bin/env python3
"""Post-result target-scope tests; no expected labels enter speech accounting."""

from copy import deepcopy

from grain_research.dialogue_speech_scope import scope_speech


def event(predicate, text):
    """Create a candidate speech observation with its original utterance."""
    return {"proposed_predicate": predicate, "utterance_span": {"quote": text}}


def test_preserve_multifunction_and_input():
    data = [
        event("let", "Let me correct my previous answer: No, I did not."),
        event("help", "Please help me with this task."),
    ]
    old = deepcopy(data)
    observed = scope_speech(data)
    assert data == old and len(observed) == 2
    assert observed[0]["explicit_previous_answer_repair"]
    assert observed[0]["discourse_permission_candidate"]
    assert not observed[0]["task_assistance_candidate"]
    assert observed[1]["task_assistance_candidate"]
    assert not observed[1]["discourse_permission_candidate"]


def test_keyword_alone_does_not_remove_help():
    observed = scope_speech([event("help", "Please help me correct my answer.")])[0]
    assert observed["task_assistance_candidate"]
    assert not observed["explicit_previous_answer_repair"]


def test_unknown_speech_retained_without_help_label():
    result = scope_speech([event("wait", "Please wait.")])
    assert len(result) == 1
    assert not result[0]["task_assistance_candidate"]
    assert not result[0]["discourse_permission_candidate"]

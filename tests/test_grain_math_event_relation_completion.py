#!/usr/bin/env python3
"""Explicit-parent amendment tests; frozen numerical procedures are unchanged."""

import ast

import pytest

from scripts.complete_grain_math_event_relation_renaming import amend_alias_parent


@pytest.mark.parametrize("reverse", [False, True])
def test_parent_lookup_does_not_depend_on_body_order(reverse):
    """An alias points to r1 even when r2 is narrated first."""
    own = [{"record_id": "r1", "event_id": "K3"}, {"record_id": "r2", "event_id": "K4"}]
    if reverse:
        own.reverse()
    x = {"parent_id": "r1", "event_id": "K3"}
    tree = amend_alias_parent(
        ast.parse('x["event_id"] == own[0]["event_id"]', mode="eval")
    )
    assert eval(compile(tree, "parent-check", "eval"), {"own": own, "x": x})


def test_unknown_parent_is_not_accepted():
    """Missing referenced parent is an error, not a permissive ownership fallback."""
    tree = amend_alias_parent(ast.parse('own[0]["event_id"]', mode="eval"))
    with pytest.raises(StopIteration):
        eval(
            compile(tree, "parent-check", "eval"),
            {"own": [{"record_id": "r2", "event_id": "K4"}], "x": {"parent_id": "r1"}},
        )


@pytest.mark.parametrize(
    "source", ['own[1]["event_id"]', 'own[0]["event_id"] + own[0]["event_id"]']
)
def test_amendment_rejects_unknown_or_multiple_repairs(source):
    """Only the identified single check may change, never arbitrary replay math."""
    with pytest.raises(ValueError):
        amend_alias_parent(ast.parse(source, mode="eval"))

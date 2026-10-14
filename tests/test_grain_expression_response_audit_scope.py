#!/usr/bin/env python3
"""Passing numerical checks cannot erase a failed frozen compatibility gate."""

from scripts.readback_grain_expression_response_hm119 import scope_status


def test_native_compatibility_failure_remains_failure():
    assert (
        scope_status(True, False)
        == "NUMERIC_AUDIT_PASSED_ORIGINAL_COMPATIBILITY_FAILED_NOT_FULL"
    )


def test_failed_numeric_audit_never_passes():
    assert scope_status(False, True) == "NUMERIC_AUDIT_FAILED"


def test_only_both_checks_get_both_passed_status():
    assert scope_status(True, True) == "NUMERIC_AUDIT_PASSED_COMPATIBILITY_PASSED"

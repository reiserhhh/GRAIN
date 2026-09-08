#!/usr/bin/env python3
"""Persisted filesystem components are not current public brand assertions."""

from scripts.audit_grain_naming import FORMER_EXPANSION, stale_brand_prose


def test_legacy_directory_is_not_brand_prose():
    assert not stale_brand_prose(f"Root: /Volumes/disk/projects/{FORMER_EXPANSION}.")
    assert not stale_brand_prose(f"Root: D:\\projects\\{FORMER_EXPANSION}\\docs")


def test_directory_exception_does_not_hide_other_stale_mentions():
    assert stale_brand_prose(f"Use /projects/{FORMER_EXPANSION}; the SUICA method.")
    assert stale_brand_prose(FORMER_EXPANSION + " is the current system.")

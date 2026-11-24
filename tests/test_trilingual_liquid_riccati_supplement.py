#!/usr/bin/env python3
"""Consistency checks for the trilingual GRAIN Version 2 liquid/Riccati supplement."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"
FILES = {
    "EN": PAPER_DIR / "GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md",
    "JA": PAPER_DIR / "GRAIN_LIQUID_RICCATI_SUPPLEMENT_JA_V2.md",
    "ZH": PAPER_DIR / "GRAIN_LIQUID_RICCATI_SUPPLEMENT_ZH_V2.md",
}
PROVENANCE = PAPER_DIR / "GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md"
KEY_NUMERIC_CLAIMS = (
    "58",
    "30.68%",
    "1.388e-17",
    "0.0216x",
    "3.8343",
    "0.3545",
    "0.5869",
    "0.4031",
    "0.7399",
    "0.2831",
    "+0.0442",
    "+0.0266",
)


def _read(language: str) -> str:
    return FILES[language].read_text(encoding="utf-8")


def test_v2_supplement_files_exist() -> None:
    assert all(path.is_file() for path in FILES.values())
    assert PROVENANCE.is_file()


def test_v2_heading_depth_and_count_are_aligned() -> None:
    signatures = []
    for language in FILES:
        headings = re.findall(r"^(#{1,3}) .+$", _read(language), re.MULTILINE)
        signatures.append(headings)
    assert signatures[0] == signatures[1] == signatures[2]
    assert len(signatures[0]) == 9


def test_v2_key_numeric_claims_are_present_in_every_version() -> None:
    for language in FILES:
        text = _read(language)
        for claim in KEY_NUMERIC_CLAIMS:
            assert claim in text, (language, claim)


def test_v2_internal_development_metaphors_absent() -> None:
    banned = {
        "EN": ("watermelon", "rind model", "flesh channel", "4W theory"),
        "JA": ("スイカの皮", "果肉チャネル", "4W理論"),
        "ZH": ("西瓜皮", "果肉通道", "4W理论"),
    }
    for language, phrases in banned.items():
        text = _read(language).lower()
        assert not any(phrase.lower() in text for phrase in phrases), language


def test_v2_evidence_provenance_is_linked_and_local_paths_exist() -> None:
    for language in FILES:
        assert "GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md" in _read(language)

    local_paths = re.findall(
        r"`((?:suica_core|grain_core|docs|tests)/[^`]+?\.(?:py|md))`",
        PROVENANCE.read_text(encoding="utf-8"),
    )
    assert local_paths
    for rel_path in local_paths:
        assert (ROOT / rel_path).is_file(), rel_path


def test_v2_supplements_state_core_nonclaim() -> None:
    required = {
        "EN": "not a personality scale",
        "JA": "パーソナリティ尺度では",
        "ZH": "不是人格量表",
    }
    for language, phrase in required.items():
        assert phrase in _read(language), language

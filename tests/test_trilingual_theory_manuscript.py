"""Consistency checks for the trilingual GRAIN theory manuscript."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"
FILES = {
    "EN": PAPER_DIR / "GRAIN_THEORY_MANUSCRIPT_EN_V1.md",
    "JA": PAPER_DIR / "GRAIN_THEORY_MANUSCRIPT_JA_V1.md",
    "ZH": PAPER_DIR / "GRAIN_THEORY_MANUSCRIPT_ZH_V1.md",
}
PROVENANCE = PAPER_DIR / "GRAIN_THEORY_EVIDENCE_PROVENANCE.md"
EXPECTED_EQUATIONS = [str(i) for i in range(1, 16)]
KEY_NUMERIC_CLAIMS = (
    "10,296",
    "17,640,062",
    "1,401",
    "8,895",
    ".069",
    "-.007",
    ".076",
    ".279",
    ".049",
)


def _read(language: str) -> str:
    return FILES[language].read_text(encoding="utf-8")


def _dois(text: str) -> set[str]:
    return {
        value.rstrip(".,)")
        for value in re.findall(r"https://doi\.org/\S+", text)
    }


def test_all_language_files_exist() -> None:
    assert all(path.is_file() for path in FILES.values())
    assert PROVENANCE.is_file()


def test_equation_numbers_are_complete_and_aligned() -> None:
    for language in FILES:
        tags = re.findall(r"\\tag\{(\d+)\}", _read(language))
        assert tags == EXPECTED_EQUATIONS, language


def test_heading_depth_and_count_are_aligned() -> None:
    signatures = []
    for language in FILES:
        headings = re.findall(r"^(#{1,3}) .+$", _read(language), re.MULTILINE)
        signatures.append(headings)
    assert signatures[0] == signatures[1] == signatures[2]
    assert len(signatures[0]) == 49


def test_reference_dois_are_identical() -> None:
    doi_sets = [_dois(_read(language)) for language in FILES]
    assert doi_sets[0] == doi_sets[1] == doi_sets[2]
    assert len(doi_sets[0]) == 23


def test_key_numeric_claims_are_present_in_every_version() -> None:
    for language in FILES:
        text = _read(language)
        for claim in KEY_NUMERIC_CLAIMS:
            assert claim in text, (language, claim)


def test_internal_development_metaphors_do_not_reenter_manuscript() -> None:
    banned = {
        "EN": ("watermelon", "rind model", "flesh channel", "4W theory"),
        "JA": ("スイカの皮", "果肉チャネル", "4W理論"),
        "ZH": ("西瓜皮", "果肉通道", "4W理论"),
    }
    for language, phrases in banned.items():
        text = _read(language).lower()
        assert not any(phrase.lower() in text for phrase in phrases), language


def test_evidence_provenance_is_linked_and_sources_exist() -> None:
    for language in FILES:
        assert "GRAIN_THEORY_EVIDENCE_PROVENANCE.md" in _read(language)

    source_paths = re.findall(r"`((?:reports|docs)/[^`]+\.md)`", PROVENANCE.read_text())
    assert source_paths
    assert all((ROOT / path).is_file() for path in source_paths)


def test_obsolete_technical_term_is_absent() -> None:
    obsolete = ("anonymous technical", "未命名の技術", "未命名技术")
    for language in FILES:
        assert not any(term in _read(language) for term in obsolete)


def test_manuscripts_state_core_nonclaim() -> None:
    required = {
        "EN": "not a personality scale",
        "JA": "パーソナリティ尺度では",
        "ZH": "不是人格量表",
    }
    for language, phrase in required.items():
        assert phrase in _read(language)

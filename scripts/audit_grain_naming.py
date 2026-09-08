#!/usr/bin/env python3
"""Inventory legacy naming and reject stale brand prose on active surfaces."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMER_EXPANSION = "Sliced Utterance In-Context Assessment"
FORMER_BRAND_RE = re.compile(r"(?<![A-Za-z0-9_./-])SUICA(?![A-Za-z0-9_/-])")
FORMER_TITLE_RE = re.compile(r"(?<![A-Za-z0-9_./-])Suica(?![A-Za-z0-9_/-])")
HISTORICAL_MARKERS = (
    "former",
    "formerly",
    "historical",
    "legacy",
    "旧称",
    "旧名",
    "旧",
    "旧システム名",
)
HISTORICAL_DOCUMENTS = {
    "docs/FREEZE_NOTES.md",
    "docs/PREREGISTRATION.md",
    "docs/RELEASE_NOTES_V020.md",
    "docs/RELEASE_NOTES_V021.md",
    "docs/SUICA_COMPLETION_DECLARATION.md",
    "docs/V7_LOCKBOX_V020.md",
}


def tracked_files() -> list[Path]:
    """Return tracked repository files without relying on filesystem noise."""
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8", errors="surrogateescape")
    return [Path(item) for item in output.split("\0") if item]


def category(path: Path) -> str:
    """Classify a path by migration obligations."""
    posix = path.as_posix()
    if posix.startswith(("release/", "reports/", "results_lockbox/")):
        return "frozen_or_persisted_evidence"
    if posix in HISTORICAL_DOCUMENTS:
        return "frozen_or_persisted_evidence"
    if posix.startswith(("configs/", "schemas/")):
        return "registered_contract_or_schema"
    if posix.startswith(("suica_core/", "suica_sim/")):
        return "legacy_python_namespace"
    if posix.startswith(("scripts/", "tests/")) or "suica" in path.name.lower():
        return "legacy_technical_identifier"
    if posix.startswith("docs/") or path.name in {
        "README.md",
        "README.zh.md",
        "README.ja.md",
        "PUBLICATION_PLAN.md",
        "CITATION.cff",
    }:
        return "active_documentation"
    return "other"


def historical_context(line: str) -> bool:
    """Return whether a former-name mention explicitly records history."""
    folded = line.casefold()
    return any(marker.casefold() in folded for marker in HISTORICAL_MARKERS)


def stale_brand_prose(line: str) -> bool:
    """Exclude a persisted directory component, but still inspect remaining prose."""
    prose = re.sub(
        r"(?<=[/\\])" + re.escape(FORMER_EXPANSION),
        "__legacy_repository_directory__",
        line,
    )
    return bool(
        FORMER_EXPANSION in prose
        or FORMER_BRAND_RE.search(prose)
        or FORMER_TITLE_RE.search(prose)
    )


def inspect() -> dict[str, object]:
    """Build a machine-readable migration inventory."""
    by_category: Counter[str] = Counter()
    by_spelling: Counter[str] = Counter()
    matching_files: set[str] = set()
    violations: list[dict[str, object]] = []

    for relative in tracked_files():
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cat = category(relative)
        file_matched = False
        for number, line in enumerate(text.splitlines(), start=1):
            spellings = {
                "former_expansion": line.count(FORMER_EXPANSION),
                "standalone_upper": len(FORMER_BRAND_RE.findall(line)),
                "standalone_title": len(FORMER_TITLE_RE.findall(line)),
                "legacy_lower_or_identifier": len(
                    re.findall(r"(?i)suica", line)
                ),
            }
            total = max(
                spellings["legacy_lower_or_identifier"],
                sum(value for key, value in spellings.items() if key != "legacy_lower_or_identifier"),
            )
            if not total:
                continue
            file_matched = True
            by_category[cat] += total
            by_spelling.update({key: value for key, value in spellings.items() if value})
            stale_brand = stale_brand_prose(line)
            if cat == "active_documentation" and stale_brand and not historical_context(line):
                violations.append(
                    {
                        "path": relative.as_posix(),
                        "line": number,
                        "text": line.strip()[:240],
                    }
                )
        if file_matched:
            matching_files.add(relative.as_posix())

    return {
        "canonical_name": "GRAIN",
        "canonical_expansion": (
            "Group-Referenced Analysis of Individual Natural-Language Expression"
        ),
        "policy": "docs/GRAIN_NAMING_MIGRATION.md",
        "matching_file_count": len(matching_files),
        "occurrences_by_category": dict(sorted(by_category.items())),
        "occurrences_by_spelling": dict(sorted(by_spelling.items())),
        "active_documentation_violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = inspect()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        output = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.strict and payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

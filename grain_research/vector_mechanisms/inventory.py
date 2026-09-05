"""Header/ID-only dataset inventory; never read private questionnaire values."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path


def inspect_csv(path: Path, *, id_field: str, count_rows=False) -> dict:
    """Inspect schema and optional ID completeness without returning any raw rows."""
    if not path.is_file():
        return {"source": path.name, "status": "MISSING"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if id_field not in fields:
            raise ValueError(f"{path.name}: missing required ID field {id_field}")
        result = {
            "source": path.name,
            "bytes": path.stat().st_size,
            "fields": fields,
            "id_field": id_field,
            "status": "HEADER_VERIFIED",
            "values_exported": False,
        }
        if count_rows:
            ids, rows, missing = set(), 0, 0
            for row in reader:
                rows += 1
                uid = row[id_field].strip()
                if uid:
                    ids.add(uid)
                else:
                    missing += 1
            result.update(rows=rows, unique_authors=len(ids), missing_id=missing)
    return result


def data_inventory(data_root: Path) -> dict:
    """Inspect only explicitly permitted PANDORA files; no recursive scale scan."""
    return {
        "pandora_profiles": inspect_csv(
            data_root / "PANDORA_official/author_profiles.csv",
            id_field="author",
            count_rows=True,
        ),
        "pandora_comments": inspect_csv(
            data_root / "PANDORA_official/all_comments_since_2015.csv",
            id_field="author",
        ),
        "big5_prepared": inspect_csv(
            data_root / "prepared/pandora_official/pandora_official_big5_prepared.csv",
            id_field="user_id",
            count_rows=True,
        ),
        "meps_policy": "NO_ACCESS_VM00_VM01; frozen no-label validation in VM06 only",
        "external_label_values_used": False,
    }


def research_source_paths(root: Path) -> list[Path]:
    """Explicit source/test/config allowlist; never include .env or raw datasets."""
    paths = sorted((root / "grain_research/vector_mechanisms").glob("*.py"))
    paths += sorted((root / "scripts").glob("run_grain_vector_mechanism*.py"))
    paths += sorted((root / "scripts").glob("summarize_grain_vector*.py"))
    paths += [
        root / name
        for name in (
            "grain_research/__init__.py",
            "suica_core/__init__.py",
            "suica_core/v7_psychometric.py",
            "suica_core/v7_geometry.py",
            "tests/test_grain_vector_mechanisms.py",
            "configs/grain_vector_contexts_v1.json",
            "configs/grain_vector_mechanisms_v1.json",
            "requirements-vector-research.txt",
        )
        if (root / name).is_file()
    ]
    return sorted(set(paths))


def code_fingerprint(root: Path) -> str:
    """Bind resumable runs to the actual observer/runner code, even off-git."""
    digest = hashlib.sha256()
    for path in research_source_paths(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def archive_source(root: Path, output: Path) -> None:
    """Preserve exact experiment code at launch; no datasets or secrets included."""
    if output.exists():
        return
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in research_source_paths(root):
            archive.write(path, path.relative_to(root).as_posix())

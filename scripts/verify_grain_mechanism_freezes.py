#!/usr/bin/env python
"""Verify recorded freeze/manifest/contract hashes of mechanism_todo_v1 packages.

Read-only over every package directory. Walks each freeze-like JSON, collects
(path -> sha256) entries in any nesting, resolves the path against the repo
root, the package directory, or the JSON file's own directory, recomputes the
hash and classifies: match / mismatch / missing / sidecar / unresolved.
AppleDouble ``._`` sidecars are volume metadata, reported but never counted
as research-file mismatches. Writes one JSON and one CSV into ``--output``,
which must not already exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "grain_vector_mechanisms" / "mechanism_todo_v1"
FREEZE_PATTERN = re.compile(
    r"(freeze|frozen|manifest|contract|delivery_manifest|artifact_manifest)"
    r"[A-Za-z0-9_\-]*\.json$"
)
HEX = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_entries(obj, key_path=()):
    """Yield (key_path, path_string, expected_sha) from any nesting."""
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and HEX.match(str(obj.get("sha256", ""))):
            yield key_path, obj["path"], obj["sha256"]
        for key, value in obj.items():
            if isinstance(value, str) and HEX.match(value) and isinstance(key, str):
                yield key_path + (key,), key, value
            elif isinstance(value, (dict, list)):
                yield from hash_entries(value, key_path + (str(key),))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from hash_entries(value, key_path + (str(index),))


def resolve(path_string: str, package_dir: Path, json_dir: Path):
    candidate = Path(path_string)
    options = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        options.extend([json_dir / candidate, package_dir / candidate, ROOT / candidate])
    for option in options:
        if option.exists() and option.is_file():
            return option
    return None


def looks_like_path(key: str) -> bool:
    return ("/" in key) or ("." in key and not key.endswith("_sha256") and key not in {"sha256"})


def verify_package(package_dir: Path):
    rows = []
    json_files = sorted(
        p for p in package_dir.rglob("*.json")
        if FREEZE_PATTERN.search(p.name) and not p.name.startswith("._")
    )
    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text())
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "package": package_dir.name, "freeze_file": str(json_path.relative_to(BASE)),
                "key_path": "", "recorded_path": "", "status": "unreadable_json",
                "expected": "", "actual": "", "detail": repr(exc)[:200],
            })
            continue
        for key_path, path_string, expected in hash_entries(payload):
            name = Path(path_string).name
            if name.startswith("._"):
                status = "sidecar"
                resolved = resolve(path_string, package_dir, json_path.parent)
                actual = sha256(resolved) if resolved else ""
                if resolved and actual != expected:
                    status = "sidecar_changed"
                elif not resolved:
                    status = "sidecar_missing"
            elif not looks_like_path(path_string):
                status, actual = "unresolved_key", ""
            else:
                resolved = resolve(path_string, package_dir, json_path.parent)
                if resolved is None:
                    status, actual = "missing", ""
                else:
                    actual = sha256(resolved)
                    status = "match" if actual == expected else "mismatch"
            rows.append({
                "package": package_dir.name,
                "freeze_file": str(json_path.relative_to(BASE)),
                "key_path": "/".join(key_path),
                "recorded_path": path_string,
                "status": status,
                "expected": expected,
                "actual": actual,
                "detail": "",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packages", nargs="*", default=None,
                        help="package directory names; default = every *_r[0-9]* directory")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    names = args.packages or sorted(
        p.name for p in BASE.iterdir()
        if p.is_dir() and not p.name.startswith("._") and re.search(r"_r\d+$|_smoke\d+$", p.name)
    )
    all_rows = []
    for name in names:
        all_rows.extend(verify_package(BASE / name))
    args.output.mkdir(parents=True)
    with open(args.output / "freeze_verification.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()) if all_rows else ["package"])
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {}
    for row in all_rows:
        bucket = summary.setdefault(row["package"], {})
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1
    problems = [r for r in all_rows if r["status"] in {"mismatch", "missing", "unreadable_json"}]
    payload = {
        "base": str(BASE.relative_to(ROOT)),
        "packages": names,
        "entries": len(all_rows),
        "summary": summary,
        "research_file_problems": problems,
        "sidecar_note": "._ AppleDouble entries are volume metadata sidecars, not research files",
    }
    (args.output / "freeze_verification.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for name in names:
        print(name, json.dumps(summary.get(name, {}), sort_keys=True))
    print("research_file_problems", len(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())

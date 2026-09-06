#!/usr/bin/env python3
"""Add a retrospective source archive without rewriting frozen experiment artifacts."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.inventory import research_source_paths


def sha(path):
    """Hash an allowlisted local source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Validate existing code digests, then supplement the narrow historical glob."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    paths = set(research_source_paths(ROOT))
    paths.update((ROOT / "scripts").glob("*grain_vector*.py"))
    paths.update((ROOT / "tests").glob("test_grain*.py"))
    for name in ("v8_bridge.py", "v8_external_connection.py"):
        paths.add(ROOT / "suica_core" / name)
    paths = sorted(
        path for path in paths if path.is_file() and not path.name.startswith("._")
    )
    records = []
    for folder in a.batch_dirs:
        manifest_path = folder / "batch_manifest.json"
        checked = 0
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for name, digest in manifest.get("code", {}).items():
                path = (ROOT / name).resolve()
                if (
                    not path.is_relative_to(ROOT)
                    or path not in paths
                    or sha(path) != digest
                ):
                    raise ValueError(
                        "Current source cannot reconstruct original manifest code"
                    )
                checked += 1
        original = folder / "source_snapshot.zip"
        with zipfile.ZipFile(original) as z:
            archived = set(z.namelist())
        records.append(
            {
                "batch": str(
                    folder.relative_to(ROOT) if folder.is_absolute() else folder
                ),
                "original_archive_sha256": sha(original),
                "original_manifest_code_digests_verified": checked,
                "added_source_paths": [
                    str(path.relative_to(ROOT))
                    for path in paths
                    if path.relative_to(ROOT).as_posix() not in archived
                ],
                "original_archive_modified": False,
            }
        )
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(path for path in out.iterdir() if not path.name.startswith("._")):
        raise ValueError("Use a fresh supplemental archive directory")
    archive = out / "source_supplement.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT).as_posix())
    result = {
        "status": "COMPLETE",
        "retrospective_supplement_not_preregistration": True,
        "git_head_at_supplement": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "source_supplement_sha256": sha(archive),
        "batches": records,
        "datasets_or_secrets_included": False,
        "scope": "Current source plus verified original manifest hashes; unbound extra files are not asserted to be historical run code",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "source_files": len(paths),
                "batches": len(records),
                "sha256": sha(archive),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

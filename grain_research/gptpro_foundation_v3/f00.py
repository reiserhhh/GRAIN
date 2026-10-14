"""F00-01 audit helpers for frozen inputs and prior finite results."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = (
    "author_id", "event_id", "episode_id", "t", "condition",
    "opportunity", "opportunity_status", "selected", "observation",
    "text_ref", "reader_id", "role",
)
FORBIDDEN_FIELDS = {"truth", "latent", "world_truth", "seed", "target_truth"}
ROLES = {"source", "validation", "application"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one declared file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_inventory(root: Path) -> list[dict]:
    """Inventory public Python functions without importing research modules."""
    rows = []
    for path in sorted(root.rglob("*.py")):
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                rows.append({
                    "path": str(path.relative_to(root)),
                    "function": node.name,
                    "line": node.lineno,
                    "docstring": bool(ast.get_docstring(node)),
                })
    return rows


def build_manifest(parent: Path, grain: Path) -> dict:
    """Build an explicit input manifest for the audit, never scanning private data."""
    source_manifest = parent / "reports/gptpro_seminar_20260915/SOURCE_MANIFEST.json"
    source_rows = json.loads(source_manifest.read_text())
    declared = [
        source_manifest,
        parent / "configs/grain_research_plan_20260915.json",
        grain / "results/grain_measurement_discovery_v1/gp01/r1/records.jsonl",
        grain / "results/grain_measurement_discovery_v1/gp02/r1/model.json",
        grain / "results/grain_measurement_discovery_v1/gp03/r1/summary.json",
        grain / "results/grain_measurement_discovery_v1/gp03/r1/metrics.csv",
        grain / "results/grain_measurement_discovery_v1/gp03/r1/manifest.json",
        grain / "grain_research/measurement_discovery/estimators.py",
        grain / "scripts/run_grain_discovery_text_channel.py",
    ]
    inputs = []
    for path in declared:
        inputs.append({
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
        })
    return {
        "schema_version": "grain-gptpro-f00-01-v3",
        "task_id": "F00-01",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_documents": source_rows,
        "inputs": inputs,
        "scope": "hash_and_replay_audit_only",
        "truth_access": "forbidden_in_analysis_helpers",
        "python": sys.version,
        "platform": platform.platform(),
    }


def verify_b1_replay(grain: Path, expected_metrics: list[dict]) -> dict:
    """Compare the cached GP03 metrics with the independent old-entry replay."""
    summary = json.loads((grain / "results/grain_measurement_discovery_v1/gp03/r1/summary.json").read_text())
    actual = summary["metrics"]
    keys = ("split", "channel", "brier", "log_loss", "events", "authors")
    same = len(actual) == len(expected_metrics) and all(
        all(left[key] == right[key] for key in keys)
        for left, right in zip(actual, expected_metrics)
    )
    oracle = {
        f"{row['split']}_{row['channel']}": row["brier"]
        for row in actual
        if row["channel"] == "oracle"
    }
    return {
        "cached_summary_matches_independent_replay": same,
        "oracle_brier": oracle,
        "metrics_rows": len(actual),
        "recomputed_prediction_file": False,
        "note": "Independent old entry point was run into a temporary directory; no old result was overwritten.",
    }


def _finite_number(value: object) -> bool:
    """Accept JSON numeric scalars but not booleans, strings or non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return bool(np.isfinite(value))


def validate_record(row: dict, splitmap: dict[str, str], seen_ids: set[str]) -> None:
    """Validate one observation row and reject ambiguity rather than repairing it."""
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    unknown_truth = sorted(FORBIDDEN_FIELDS.intersection(row))
    if unknown_truth:
        raise ValueError(f"truth-bearing fields are forbidden: {unknown_truth}")
    for field in ("author_id", "event_id", "episode_id", "text_ref", "reader_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{field} must be a non-empty string")
    event_id = row["event_id"]
    if event_id in seen_ids:
        raise ValueError(f"duplicate event_id: {event_id}")
    seen_ids.add(event_id)
    role = row["role"]
    if role not in ROLES or row["author_id"] not in splitmap:
        raise ValueError("role or splitmap is missing")
    if splitmap[row["author_id"]] != role:
        raise ValueError("row.role disagrees with splitmap")
    t = row["t"]
    if t is not None and not _finite_number(t):
        raise ValueError("t must be null or a finite numeric value")
    condition = row["condition"]
    if not isinstance(condition, dict) or not condition:
        raise ValueError("condition must be a non-empty JSON object")
    if any(not isinstance(key, str) or not _finite_number(value) for key, value in condition.items()):
        raise ValueError("condition values must be finite numeric scalars")
    observation = row["observation"]
    if observation is not None and not _finite_number(observation):
        raise ValueError("observation must be null or a finite numeric value")
    if not isinstance(row["opportunity_status"], str):
        raise ValueError("opportunity_status must be a string")


def validate_manifest(manifest: dict, expected: dict | None = None) -> None:
    """Validate the metadata contract used when joining versioned artifacts."""
    required = {"schema_version", "target_id", "world_id", "reader_id", "estimator_version", "parent_hash", "role_counts"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest missing: {missing}")
    if not all(isinstance(manifest[key], str) and manifest[key] for key in required - {"role_counts"}):
        raise ValueError("manifest identity fields must be non-empty strings")
    if not isinstance(manifest["role_counts"], dict) or set(manifest["role_counts"]) != ROLES:
        raise ValueError("role_counts must contain exactly the three roles")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in manifest["role_counts"].values()):
        raise ValueError("role_counts must be non-negative integers")
    if expected:
        for key in ("target_id", "world_id", "reader_id", "estimator_version"):
            if key in expected and manifest[key] != expected[key]:
                raise ValueError(f"manifest mismatch: {key}")
        if "parent_hash" in expected and manifest["parent_hash"] != expected["parent_hash"]:
            raise ValueError("manifest mismatch: parent_hash")


def join_artifact(records_manifest: dict, model_manifest: dict) -> dict:
    """Join only matching target/world/reader/estimator contracts."""
    validate_manifest(records_manifest)
    validate_manifest(model_manifest, records_manifest)
    return {
        "schema_version": "grain-gptpro-joined-v3",
        "target_id": records_manifest["target_id"],
        "world_id": records_manifest["world_id"],
        "reader_id": records_manifest["reader_id"],
        "estimator_version": model_manifest["estimator_version"],
        "parent_hash": model_manifest["parent_hash"],
        "role_counts": records_manifest["role_counts"],
    }

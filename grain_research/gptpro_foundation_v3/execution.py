"""Immutable execution requests and revision-aware dependency verification."""

from __future__ import annotations

import json
from pathlib import Path

from .acceptance import verify_evidence
from .f00 import sha256


def load_execution_request(path: Path, task: dict, root: Path, protocol: Path) -> dict:
    """Read, never manufacture, a request with explicit dependency revisions.

    Dependency summaries must have separately reviewed implementation and protocol
    statuses. Neither this loader nor a runner may infer review from file presence.
    """
    request = json.loads(path.read_text())
    if request.get("schema_version") != "grain-execution-request-v1":
        raise ValueError("unsupported execution request schema")
    if request.get("task_id") != task["id"]:
        raise ValueError("task identity mismatch")
    if request.get("protocol_sha256") != sha256(protocol):
        raise ValueError("frozen protocol hash mismatch")
    dependencies = request.get("dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != set(
        task["depends_on"]
    ):
        raise ValueError("dependency IDs must exactly match the frozen DAG")
    for dep_id, entry in dependencies.items():
        errors = verify_evidence([entry], root)
        if errors:
            raise ValueError(f"{dep_id}: {errors}")
        manifest_path = (root / entry["path"]).resolve()
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("task_id") != dep_id:
            raise ValueError(f"dependency task identity mismatch: {dep_id}")
        errors = verify_evidence(manifest.get("files"), manifest_path.parent)
        if errors:
            raise ValueError(f"{dep_id} artifact integrity: {errors}")
        if not any(file["path"] == "summary.json" for file in manifest["files"]):
            raise ValueError(f"summary is not bound by manifest: {dep_id}")
        summary = json.loads((manifest_path.parent / "summary.json").read_text())
        if summary.get("task_id") != dep_id:
            raise ValueError(f"summary task identity mismatch: {dep_id}")
        if any(
            summary.get(key) != "PASS"
            for key in ("implementation_status", "protocol_status")
        ):
            raise ValueError(
                f"dependency not implementation/protocol accepted: {dep_id}"
            )
        review = entry.get("acceptance")
        if not isinstance(review, dict) or review.get("status") != "ACCEPTED":
            raise ValueError(f"missing dependency review: {dep_id}")
        if not review.get("reviewer") or not review.get("reason"):
            raise ValueError(f"incomplete dependency review: {dep_id}")
    return request


def fresh_output(path: Path, input_path: Path) -> Path:
    """Reject old result directories and input paths inside the write area."""
    path = path.resolve()
    if input_path.resolve().is_relative_to(path):
        raise ValueError("input manifest must be outside the output directory")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"existing results must not be overwritten: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path

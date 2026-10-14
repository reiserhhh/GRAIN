"""Mutation tests for versioned execution dependencies."""

import json

import pytest

from grain_research.gptpro_foundation_v3.execution import (
    fresh_output,
    load_execution_request,
)
from grain_research.gptpro_foundation_v3.f00 import sha256


def fixture_request(tmp_path):
    """Make a tiny explicitly reviewed dependency chain for loader tests."""
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen test specification")
    dep = tmp_path / "r2"
    dep.mkdir()
    summary = dep / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "task_id": "F00-01",
                "implementation_status": "PASS",
                "protocol_status": "PASS",
            }
        )
    )
    manifest = dep / "output_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "F00-01",
                "files": [{"path": "summary.json", "sha256": sha256(summary)}],
            }
        )
    )
    request = {
        "schema_version": "grain-execution-request-v1",
        "task_id": "F00-02",
        "protocol_sha256": sha256(protocol),
        "dependencies": {
            "F00-01": {
                "path": "r2/output_manifest.json",
                "sha256": sha256(manifest),
                "acceptance": {
                    "status": "ACCEPTED",
                    "reviewer": "test",
                    "reason": "test only",
                },
            }
        },
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return path, protocol, request, {"id": "F00-02", "depends_on": ["F00-01"]}


def test_request_read_only_and_revision_aware(tmp_path):
    path, protocol, request, task = fixture_request(tmp_path)
    digest = sha256(path)
    assert load_execution_request(path, task, tmp_path, protocol) == request
    assert sha256(path) == digest
    (tmp_path / "r2/summary.json").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        load_execution_request(path, task, tmp_path, protocol)


@pytest.mark.parametrize(
    "mutation", ["missing_dependency", "wrong_protocol", "wrong_task", "no_review"]
)
def test_bad_request_rejected(tmp_path, mutation):
    path, protocol, request, task = fixture_request(tmp_path)
    if mutation == "missing_dependency":
        request["dependencies"] = {}
    elif mutation == "wrong_protocol":
        request["protocol_sha256"] = "0" * 64
    elif mutation == "wrong_task":
        request["task_id"] = "F00-03"
    else:
        request["dependencies"]["F00-01"].pop("acceptance")
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        load_execution_request(path, task, tmp_path, protocol)


def test_output_cannot_clobber_results_or_inputs(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "old.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        fresh_output(output, tmp_path / "request.json")
    with pytest.raises(ValueError):
        fresh_output(tmp_path / "new", tmp_path / "new/input.json")

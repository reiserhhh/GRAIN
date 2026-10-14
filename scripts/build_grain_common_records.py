#!/usr/bin/env python
"""F00 common-record adapter for seminar-06 session packets (engineering integration r1).

Part A maps the real C2 comment input to F00 common records (fold 0, np_mcd author set) and validates every row
with `f00.validate_record`, rejecting rather than repairing. Violations are tallied by type. It then checks the
records/model manifests with `f00.validate_manifest` and `f00.join_artifact`.

Part B writes the F00-04 eight required artifacts for each session packet into `<packet>/common_record_r1/`,
derived from existing files only. Nothing is rerun, and packet outputs are never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grain_research.gptpro_foundation_v3 import f00  # noqa: E402

RES = REPO / "results/grain_gptpro_foundation_v3"
OUT = RES / "F00/common_record_adapter_r1"
PY = "/Volumes/mobile3/projects/project persona/.venv/bin/python"
REQUIRED = ["protocol.json", "input_manifest.json", "summary.json", "metrics.csv", "decision.md", "run_log.jsonl", "tests.txt", "output_manifest.json"]
PACKETS = {
    "P05/native_action_r1": {"protocols": ["docs/GRAIN_P05_NATIVE_ACTION_PACKET_V1.md"], "state": "COUNTEREXAMPLE",
                             "headline": "action-weighted (J) estimator rejected at frozen scope: J24-E24 dr -.29 [-.48,-.07]; no cell beats zero RMSE",
                             "model_calls": "956 passes (packet accounting)", "tests": ["tests/test_grain_native_action.py"]},
    "P05/native_geometry_r1": {"protocols": ["docs/GRAIN_P05_NATIVE_GEOMETRY_DIAGNOSTIC_V1.md"], "state": "PASS",
                               "headline": "source-gradient LOO span coverage .973/.981; donor energy in span 2-5% (random .005); descriptive, J branch pending",
                               "model_calls": "0", "tests": []},
    "F12/criterion_preservation_r1": {"protocols": ["docs/GRAIN_CRITERION_PRESERVATION_PROTOCOL_V1.md"], "state": "INCONCLUSIVE",
                                      "headline": "topic-residualized E criterion r .61 -> .58 (-.03 [-.11,.04]); interval includes 0, not equivalence",
                                      "model_calls": "0", "tests": []},
    "P05/c2_venue_dependence_r1": {"protocols": ["docs/GRAIN_C2_MCD_VENUE_DEPENDENCE_PROTOCOL_V1.md", "docs/GRAIN_C2_MCD_VENUE_DEPENDENCE_PROTOCOL_V1_A1.md"],
                                   "state": "COUNTEREXAMPLE",
                                   "headline": "counterexample to a general MCD increment: B-C -.0244 [-.034,-.016]; B-LRNP -.0075; replay gate FAIL (tau) -> variant c2_v1-local",
                                   "model_calls": "0 (local CPU Ridge/SVD/k-means)", "tests": ["tests/test_grain_c2_venue_dependence.py"]},
    "F17/h01_person_n_recovery_r1": {"protocols": ["docs/GRAIN_H01_PERSON_N_RECOVERY_PROTOCOL_V1.md"], "state": "PASS",
                                     "headline": "H01 N 300 under final design (seminar05_item4 Part B power .845); design input, not validation",
                                     "model_calls": "0", "tests": []},
    "F17/h02_recovery_r1": {"protocols": ["docs/GRAIN_H02_RECOVERY_PROTOCOL_V1.md", "docs/GRAIN_H02_RECOVERY_PROTOCOL_V1_A1.md"], "state": "INCONCLUSIVE",
                            "headline": "rival-ID N 150/450/600; criterion tetrad underpowered <=600 so frozen rule yields no N; population-level only",
                            "model_calls": "0", "tests": []},
}


def sha(path: Path) -> str:
    return f00.sha256(path)


def files_of(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file() and not p.name.startswith("._") and "common_record_r1" not in p.parts
                  and "mcd_cache" not in p.parts and "cells" not in p.parts)


def part_a() -> dict:
    """Common records for the C2 input, fold 0 / np_mcd author set."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("c2", REPO / "scripts/run_grain_c2_mcd_venue_dependence.py")
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    comments = pd.read_csv(c2.COMMENTS, usecols=["user_id", "comment_id", "created_utc", "subreddit", "link_id"])
    comments["user_id"] = comments["user_id"].astype(str)
    artifacts = c2.orig.load_baseline_artifacts(c2.BASELINE)
    frame, _v, _p = c2.orig.trait_frame(artifacts, "agreeableness", 0, True)
    tr, va, te = c2.orig.split_indices(frame, 0)
    authors = frame["author"].astype(str).to_numpy()
    splitmap = {**{a: "source" for a in authors[tr]}, **{a: "validation" for a in authors[va]}, **{a: "application" for a in authors[te]}}
    seen: set[str] = set()
    errors: Counter = Counter()
    first_examples: dict[str, str] = {}
    role_counts = Counter()
    for row in comments.itertuples(index=False):
        rec = {
            "author_id": row.user_id, "event_id": f"{row.user_id}|{row.comment_id}",
            "episode_id": str(row.link_id) if isinstance(row.link_id, str) and row.link_id else "",
            "t": float(row.created_utc) if pd.notna(row.created_utc) else None,
            "condition": {"trait_named_venue": float(c2.is_trait_named(row.subreddit))},
            "opportunity": None, "opportunity_status": "unobserved_no_exposure_denominator",
            "selected": True, "observation": None, "text_ref": f"comment:{row.comment_id}",
            "reader_id": "mcd_c2_v1-local", "role": splitmap.get(row.user_id, "unassigned"),
        }
        try:
            f00.validate_record(rec, splitmap, seen)
            role_counts[rec["role"]] += 1
        except ValueError as exc:
            kind = str(exc).split(":")[0]
            errors[kind] += 1
            first_examples.setdefault(kind, rec["event_id"])
    records_manifest = {"schema_version": "grain-gptpro-records-v3", "target_id": "pandora_big5_agreeableness_np_mcd_fold0",
                        "world_id": "pandora_official_big5_repeat0", "reader_id": "mcd_c2_v1-local", "estimator_version": "c2_v1-local",
                        "parent_hash": sha(c2.COMMENTS), "role_counts": {r: int(role_counts.get(r, 0)) for r in f00.ROLES}}
    model_manifest = dict(records_manifest)
    f00.validate_manifest(records_manifest)
    joined = f00.join_artifact(records_manifest, model_manifest)
    try:
        f00.join_artifact(records_manifest, {**model_manifest, "reader_id": "other_reader"})
        mismatch_rejected = False
    except ValueError:
        mismatch_rejected = True
    dup = comments.duplicated(["user_id", "comment_id"], keep="first")
    trait = comments["subreddit"].map(c2.is_trait_named)
    return {
        "rows": int(len(comments)), "valid_records": int(sum(role_counts.values())), "role_counts": dict(role_counts),
        "violations": dict(errors), "violation_examples": first_examples,
        "record_contract_status": "FAIL (rejected, not repaired)" if errors else "PASS",
        "duplicate_rows": int(dup.sum()), "duplicate_rows_in_trait_named_venues": int((dup & trait).sum()),
        "trait_named_share_all_rows": float(trait.mean()), "trait_named_share_duplicate_rows": float(trait[dup].mean()),
        "records_manifest": records_manifest, "join_ok": joined["estimator_version"] == "c2_v1-local", "reader_mismatch_rejected": mismatch_rejected,
    }


def run_tests(paths: list[str]) -> str:
    if not paths:
        return "no packet-specific unit tests (simulation/analysis packet); acceptance rests on frozen protocol + recorded outputs\n"
    res = subprocess.run([PY, "-m", "pytest", "-q", *paths], cwd=REPO, capture_output=True, text=True)
    return res.stdout[-600:] + "\n"


def part_b(started: str) -> list[dict]:
    out = []
    for rel, spec in PACKETS.items():
        d = RES / rel
        cr = d / "common_record_r1"
        cr.mkdir(exist_ok=True)
        originals = files_of(d)
        protos = [{"path": p, "sha256": sha(REPO / p)} for p in spec["protocols"]]
        state = spec["state"]
        assert state in {"PLANNED", "BUILT", "TESTED", "RUNNING", "PASS", "COUNTEREXAMPLE", "INCONCLUSIVE", "BLOCKED_INPUT"}
        (cr / "protocol.json").write_text(json.dumps({"packet": rel, "protocol_docs": protos, "state": state,
            "application_rule": "application cannot update source fit or select model", "private_data": False,
            "neural_training": False, "model_calls": spec["model_calls"]}, indent=1) + "\n")
        (cr / "input_manifest.json").write_text(json.dumps({"packet": rel, "packet_files": [{"path": str(p.relative_to(d)), "sha256": sha(p)} for p in originals]}, indent=1) + "\n")
        (cr / "summary.json").write_text(json.dumps({"packet": rel, "status": state, "headline": spec["headline"],
            "decision": f"{rel}/decision.md", "promotion": "none (awaits batch re-audit)"}, indent=1) + "\n")
        with (cr / "metrics.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["packet", "check", "status", "value"])
            w.writeheader()
            w.writerow({"packet": rel, "check": "packet_decision_present", "status": "PASS" if (d / "decision.md").exists() else "FAIL", "value": ""})
            w.writerow({"packet": rel, "check": "protocol_docs_hashed", "status": "PASS", "value": len(protos)})
            w.writerow({"packet": rel, "check": "packet_files_hashed", "status": "PASS", "value": len(originals)})
        (cr / "decision.md").write_text(f"# {rel} common record\n\nState: {state}. {spec['headline']}\n\nThe authoritative decision is `{rel}/decision.md` (hashed in input_manifest.json). This record adds no new result.\n")
        (cr / "run_log.jsonl").write_text(json.dumps({"packet": rel, "adapter_started": started, "adapter_finished": datetime.now(timezone.utc).isoformat(), "adapter_model_calls": 0}) + "\n")
        (cr / "tests.txt").write_text(run_tests(spec["tests"]))
        (cr / "output_manifest.json").write_text(json.dumps({"packet": rel, "status": state, "files": [{"path": p.name, "sha256": sha(p)} for p in sorted(cr.iterdir()) if p.name != "output_manifest.json" and not p.name.startswith("._")]}, indent=1) + "\n")
        present = [f for f in REQUIRED if (cr / f).exists()]
        manifest = json.loads((cr / "output_manifest.json").read_text())
        hashes_ok = all(sha(cr / e["path"]) == e["sha256"] for e in manifest["files"])
        out.append({"packet": rel, "state": state, "required_present": len(present), "required_total": len(REQUIRED), "hashes_ok": hashes_ok, "packet_files": len(originals)})
    return out


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    a = part_a()
    b = part_b(started)
    report = {"part_a_c2_common_records": a, "part_b_eight_artifact_records": b, "adapter_sha256": sha(Path(__file__)), "started": started}
    (OUT / "adapter_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

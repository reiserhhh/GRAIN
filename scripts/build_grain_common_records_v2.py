#!/usr/bin/env python
"""F00 common-record adapter v2 (seminar-07 §3). v1 and its r1 outputs are left unchanged.

Fixes relative to v1:
- A packet's test subprocess must exit 0. A nonzero exit raises PackagingError, and no output_manifest is written
  for that packet, so a failed child cannot look like a successful package.
- C2 input records are built per condition (np_mcd, mcd_only). Rows of authors outside a condition's frame are
  reported as condition-ineligible exclusions with a reason, never given a dummy role. Duplicate events are still
  rejected, and the c2_dedup inventory classes are attached.
- Records are written for the new seminar-07 packets. Existing packet records are not touched.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grain_research.gptpro_foundation_v3 import f00  # noqa: E402

RES = REPO / "results/grain_gptpro_foundation_v3"
OUT = RES / "F00/common_record_adapter_r2"
PY = "/Volumes/mobile3/projects/project persona/.venv/bin/python"
REQUIRED = ["protocol.json", "input_manifest.json", "summary.json", "metrics.csv", "decision.md", "run_log.jsonl", "tests.txt", "output_manifest.json"]
STATES = {"PLANNED", "BUILT", "TESTED", "RUNNING", "PASS", "COUNTEREXAMPLE", "INCONCLUSIVE", "BLOCKED_INPUT"}
PACKETS = {
    "P05/c2_dedup_sensitivity_r1": {"protocols": ["docs/GRAIN_C2_DEDUP_SENSITIVITY_PROTOCOL_V1.md"], "state": "BLOCKED_INPUT",
        "headline": "stopped at inventory by frozen rule: 1,795 duplicate-key rows conflict on consumed fields; no fits",
        "model_calls": "0", "tests": ["tests/test_grain_c2_venue_dependence.py"]},
    "R14/counterfactual_reader_r1": {"protocols": ["docs/GRAIN_COUNTERFACTUAL_READER_PACKET_V1.md", "docs/GRAIN_COUNTERFACTUAL_READER_PACKET_V1_A1.md"],
        "state": "PASS", "headline": "680/680 calls; Qwen E+ +.331, E- -.118, ownership -.046; L12.H0 transfer E- -.045 vs control -.0007",
        "model_calls": "680 local sample-forwards: Qwen 480 (pilot 96 + full 384), NLI 200; see accounting.jsonl", "tests": []},
    "F17/h02_recovery_r2": {"protocols": ["docs/GRAIN_H02_RECOVERY_PROTOCOL_V2.md"], "state": "PASS",
        "headline": "encompassing mechanism-set rule: identification-component design N 450 at d .35 (simulation)",
        "model_calls": "0", "tests": []},
}


class PackagingError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return f00.sha256(path)


def run_tests(paths: list[str]) -> tuple[bool, str]:
    if not paths:
        return True, "no packet-specific unit tests; acceptance rests on the frozen protocol and recorded outputs\n"
    res = subprocess.run([PY, "-m", "pytest", "-q", *paths], cwd=REPO, capture_output=True, text=True)
    text = f"exit_code={res.returncode}\n" + res.stdout[-800:] + res.stderr[-400:] + "\n"
    return res.returncode == 0, text


def files_of(d: Path) -> list[Path]:
    skip = {"common_record_r1", "mcd_cache", "cells", "private_tensors"}
    return sorted(p for p in d.rglob("*") if p.is_file() and not p.name.startswith("._") and not skip.intersection(p.parts))


def build_record(rel: str, spec: dict, started: str, root: Path = RES) -> dict:
    d = root / rel
    cr = d / "common_record_r1"
    if spec["state"] not in STATES:
        raise PackagingError(f"state not allowed: {spec['state']}")
    ok, test_text = run_tests(spec["tests"])
    cr.mkdir(exist_ok=True)
    (cr / "tests.txt").write_text(test_text)
    if not ok:
        raise PackagingError(f"{rel}: packet tests failed; no output_manifest written")
    originals = files_of(d)
    protos = [{"path": p, "sha256": sha(REPO / p)} for p in spec["protocols"]]
    (cr / "protocol.json").write_text(json.dumps({"packet": rel, "protocol_docs": protos, "state": spec["state"],
        "application_rule": "application cannot update source fit or select model", "private_data": False,
        "neural_training": False, "model_calls": spec["model_calls"], "adapter_version": 2}, indent=1) + "\n")
    (cr / "input_manifest.json").write_text(json.dumps({"packet": rel, "packet_files": [{"path": str(p.relative_to(d)), "sha256": sha(p)} for p in originals]}, indent=1) + "\n")
    (cr / "summary.json").write_text(json.dumps({"packet": rel, "status": spec["state"], "headline": spec["headline"],
        "decision": f"{rel}/decision.md", "promotion": "none (awaits source-specific review)"}, indent=1) + "\n")
    with (cr / "metrics.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["packet", "check", "status", "value"])
        w.writeheader()
        w.writerow({"packet": rel, "check": "packet_decision_present", "status": "PASS" if (d / "decision.md").exists() else "FAIL", "value": ""})
        w.writerow({"packet": rel, "check": "packet_tests_exit_zero", "status": "PASS", "value": len(spec["tests"])})
        w.writerow({"packet": rel, "check": "packet_files_hashed", "status": "PASS", "value": len(originals)})
    (cr / "decision.md").write_text(f"# {rel} common record (adapter v2)\n\nState: {spec['state']}. {spec['headline']}\n\nThe authoritative decision is `{rel}/decision.md`. This record adds no new result, and serialization does not close the science.\n")
    (cr / "run_log.jsonl").write_text(json.dumps({"packet": rel, "adapter_started": started, "adapter_finished": datetime.now(timezone.utc).isoformat(), "adapter_model_calls": 0}) + "\n")
    (cr / "output_manifest.json").write_text(json.dumps({"packet": rel, "status": spec["state"], "files": [{"path": p.name, "sha256": sha(p)} for p in sorted(cr.iterdir()) if p.name != "output_manifest.json" and not p.name.startswith("._")]}, indent=1) + "\n")
    present = [f for f in REQUIRED if (cr / f).exists()]
    m = json.loads((cr / "output_manifest.json").read_text())
    return {"packet": rel, "state": spec["state"], "required_present": len(present), "hashes_ok": all(sha(cr / e["path"]) == e["sha256"] for e in m["files"])}


def c2_records() -> dict:
    spec = importlib.util.spec_from_file_location("c2", REPO / "scripts/run_grain_c2_mcd_venue_dependence.py")
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    comments = pd.read_csv(c2.COMMENTS, usecols=["user_id", "comment_id", "created_utc", "subreddit", "link_id"])
    comments["user_id"] = comments["user_id"].astype(str)
    artifacts = c2.orig.load_baseline_artifacts(c2.BASELINE)
    pdf = artifacts["p_features"]
    out = {}
    for cond, use_p in (("np_mcd", True), ("mcd_only", False)):
        frame, _v, _p = c2.orig.trait_frame(artifacts, "agreeableness", 0, use_p)
        tr, va, te = c2.orig.split_indices(frame, 0)
        a = frame["author"].astype(str).to_numpy()
        splitmap = {**{x: "source" for x in a[tr]}, **{x: "validation" for x in a[va]}, **{x: "application" for x in a[te]}}
        eligible = comments["user_id"].isin(splitmap)
        excluded = comments.loc[~eligible, "user_id"]
        p_ok = set(pdf["author"].astype(str))
        reasons = Counter("no_P_features" if (use_p and u not in p_ok) else "outside_frame_other" for u in excluded)
        seen, errors, roles = set(), Counter(), Counter()
        for row in comments[eligible].itertuples(index=False):
            rec = {"author_id": row.user_id, "event_id": f"{row.user_id}|{row.comment_id}", "episode_id": str(row.link_id),
                   "t": float(row.created_utc), "condition": {"trait_named_venue": float(c2.is_trait_named(row.subreddit))},
                   "opportunity": None, "opportunity_status": "unobserved_no_exposure_denominator", "selected": True,
                   "observation": None, "text_ref": f"comment:{row.comment_id}", "reader_id": "mcd_c2_v1-local", "role": splitmap[row.user_id]}
            try:
                f00.validate_record(rec, splitmap, seen)
                roles[rec["role"]] += 1
            except ValueError as exc:
                errors[str(exc).split(":")[0]] += 1
        out[cond] = {"eligible_rows": int(eligible.sum()), "excluded_rows": int((~eligible).sum()),
                     "excluded_authors": int(excluded.nunique()), "exclusion_reasons_rows": dict(reasons),
                     "valid_records": int(sum(roles.values())), "role_counts": dict(roles), "violations": dict(errors),
                     "record_contract_status": "FAIL (duplicate events rejected, not repaired)" if errors else "PASS"}
    inv = RES / "P05/c2_dedup_sensitivity_r1/inventory.json"
    out["dedup_inventory_classes"] = json.loads(inv.read_text())["classes_of_extra_rows"] if inv.exists() else None
    return out


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"c2_condition_records": c2_records(), "records": [], "packaging_failures": []}
    for rel, spec in PACKETS.items():
        try:
            report["records"].append(build_record(rel, spec, started))
        except PackagingError as exc:
            report["packaging_failures"].append(str(exc))
    report["adapter_sha256"] = sha(Path(__file__))
    (OUT / "adapter_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 1 if report["packaging_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

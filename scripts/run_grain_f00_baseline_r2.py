#!/usr/bin/env python
"""F00-01 current revision (packet 13 Part A4): source/plan hash inventory, legacy root inventory, function inventory,
an ACTUAL GP03 B1 replay into a fresh directory, and a corrected audit. CPU only, 0 neural calls.

The rejected legacy F00-01 (F00/r1/F00-01) compared the cached GP03 summary with literal numbers typed into its
script. Its summary says an independent run happened, but the code runs nothing (`recomputed_prediction_file: false`).
This revision re-runs the existing GP03 entry point on the archived GP01 records and GP02 model, and compares
byte-for-byte. Old results are read only.
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402
from grain_research.gptpro_foundation_v3.f00 import function_inventory  # noqa: E402

PERSONA = Path("/Volumes/mobile3/projects/project persona")
OUT = rs.RES / "F00/baseline_inventory_r2"
LEGACY = REPO / "results/grain_measurement_discovery_v1"
SNAPSHOT = PERSONA / "reports/gptpro_seminar_20260915/full_coverage_audit/frozen_artifacts_snapshot.json"
LEGACY_LITERAL_EXPECTED = [  # the literal list embedded in scripts/run_grain_gptpro_f00.py (legacy F00-01)
    {"split": "validation", "channel": "oracle", "brier": 0.19343257893885188, "log_loss": 0.57542122970146, "events": 3600, "authors": 100},
    {"split": "validation", "channel": "text_soft", "brier": 0.24112330416013153, "log_loss": 0.6751885908270995, "events": 3600, "authors": 100},
    {"split": "validation", "channel": "text_hard", "brier": 0.23230397188036953, "log_loss": 0.6606925741727397, "events": 3600, "authors": 100},
    {"split": "application", "channel": "oracle", "brier": 0.18910631495521518, "log_loss": 0.565869766557999, "events": 7200, "authors": 200},
    {"split": "application", "channel": "text_soft", "brier": 0.24356410625235037, "log_loss": 0.6804650094018099, "events": 7200, "authors": 200},
    {"split": "application", "channel": "text_hard", "brier": 0.22684462906230446, "log_loss": 0.6486748301641344, "events": 7200, "authors": 200},
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sources() -> dict:
    man = json.loads((PERSONA / "reports/gptpro_seminar_20260915/SOURCE_MANIFEST.json").read_text())
    rows = []
    for s in man:
        p = PERSONA / "gptproplans" / s["file"]
        rows.append({"id": s["id"], "file": s["file"], "recorded_sha256": s["sha256"], "exists": p.exists(),
                     "current_sha256": sha(p) if p.exists() else None, "match": p.exists() and sha(p) == s["sha256"]})
    plans = {"old_machine_plan": {"path": "configs/grain_research_plan_20260915.json", "recorded_sha256": "52c1399ac9beae7563eb156da3f67a20f989d77b93497a2e097a19b6ec657a7c",
                                  "recorded_in": "reports/gptpro_seminar_20260915/COMPREHENSIVE_ACCEPTANCE_AUDIT.md; full_coverage_audit/source_hash_verification.json"},
             "atomic_v3_plan": {"path": "configs/grain_gptpro_atomic_v3.json", "recorded_sha256": "1da33ffbc87259c4a8de8dff64f80a6e6e25346a9314985e0e8fbe9694e8c1bb",
                                "recorded_in": "F00/stage_orchestrator_r1/stage_integrity_summary.json"}}
    for v in plans.values():
        p = PERSONA / v["path"]
        v["current_sha256"] = sha(p) if p.exists() else None
        v["match"] = v["current_sha256"] == v["recorded_sha256"]
    return {"sources": rows, "plans": plans, "all_match": all(r["match"] for r in rows) and all(v["match"] for v in plans.values())}


def legacy_roots() -> dict:
    snap = json.loads(SNAPSHOT.read_text())
    rows = []
    for key, digest in sorted(snap.items()):
        p = REPO / key
        rows.append({"file": key, "snapshot_20260915_sha256": digest, "exists": p.exists(), "match": p.exists() and sha(p) == digest})
    reg = rs.load_registry()
    legacy_cards = {"roots": 0, "integrity": {}}
    for k, e in reg["revisions"].items():
        if e["kind"] == "REJECTED_LEGACY":
            st = rs.integrity(k, e)["integrity"]
            legacy_cards["roots"] += 1
            legacy_cards["integrity"][st] = legacy_cards["integrity"].get(st, 0) + 1
    return {"gp_pa_pv_files_vs_20260915_snapshot": {"files": len(rows), "match": sum(r["match"] for r in rows),
                                                     "mismatch_or_missing": [r["file"] for r in rows if not r["match"]]},
            "gp_pa_pv_rows": rows, "rejected_legacy_card_roots": legacy_cards,
            "snapshot_note": "the 2026-09-15 snapshot is untracked in git; a match shows unchanged bytes since that snapshot, not immutability since creation"}


def b1_replay() -> dict:
    gp03 = LEGACY / "gp03/r1"
    work = OUT / "b1_replay"
    cmd = [sys.executable, "scripts/run_grain_discovery_text_channel.py", "--input", str(LEGACY / "gp01/r1/records.jsonl"),
           "--model", str(LEGACY / "gp02/r1/model.json"), "--output", str(work)]
    before = {f: sha(gp03 / f) for f in ("metrics.csv", "summary.json", "text_channel_scores.csv", "reader.joblib")}
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(REPO), "HOME": str(Path.home())})
    after = {f: sha(gp03 / f) for f in before}
    new_metrics = json.loads((work / "summary.json").read_text())["metrics"] if (work / "summary.json").exists() else None
    cached = json.loads((gp03 / "summary.json").read_text())["metrics"]
    keys = ("split", "channel", "brier", "log_loss", "events", "authors")
    same = lambda a, b: a is not None and len(a) == len(b) and all(all(x[k] == y[k] for k in keys) for x, y in zip(a, b))  # noqa: E731
    gp02 = json.loads((LEGACY / "gp02/r1/model.json").read_text())
    res = {"command": " ".join(cmd[1:]), "exit": done.returncode, "stderr_tail": done.stderr[-600:] if done.returncode else "",
           "metrics_csv_byte_identical": (work / "metrics.csv").exists() and filecmp.cmp(work / "metrics.csv", gp03 / "metrics.csv", shallow=False),
           "scores_csv_byte_identical": (work / "text_channel_scores.csv").exists() and filecmp.cmp(work / "text_channel_scores.csv", gp03 / "text_channel_scores.csv", shallow=False),
           "reader_joblib_byte_identical": (work / "reader.joblib").exists() and filecmp.cmp(work / "reader.joblib", gp03 / "reader.joblib", shallow=False),
           "summary_metrics_equal_cached": same(new_metrics, cached), "cached_equal_legacy_literal": same(cached, LEGACY_LITERAL_EXPECTED),
           "b1_fixed_source_table": gp02["b1"]["source_p"], "oracle_brier": {f"{m['split']}": m["brier"] for m in (new_metrics or []) if m["channel"] == "oracle"},
           "legacy_gp03_unchanged": before == after}
    res["status"] = "REPLAYED_EXACT" if (res["exit"] == 0 and res["metrics_csv_byte_identical"] and res["scores_csv_byte_identical"]
                                          and res["summary_metrics_equal_cached"] and res["legacy_gp03_unchanged"]) else "REPLAY_DIFFERS"
    return res


def main() -> int:
    if OUT.exists():
        raise FileExistsError("baseline_inventory_r2 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    t0 = datetime.now(timezone.utc)
    src = sources()
    leg = legacy_roots()
    inv = {"measurement_discovery": function_inventory(REPO / "grain_research/measurement_discovery"),
           "gptpro_foundation_v3_public_function_count": len(function_inventory(REPO / "grain_research/gptpro_foundation_v3"))}
    rep = b1_replay()
    baseline = {"task_id": "F00-01", "revision": "F00/baseline_inventory_r2", "created_at": t0.isoformat(), "source_and_plan_hashes": src,
                "legacy_roots": {k: v for k, v in leg.items() if k != "gp_pa_pv_rows"},
                "function_inventory": {"measurement_discovery_public_functions": len(inv["measurement_discovery"]),
                                       "gptpro_foundation_v3_public_functions": inv["gptpro_foundation_v3_public_function_count"],
                                       "note": "AST inventory of existing functions; nothing imported or fabricated"},
                "b1_replay": rep, "missing_sources": [r["id"] for r in src["sources"] if not r["exists"]],
                "private_data": False, "neural_calls": 0}
    (OUT / "baseline_inventory.json").write_text(json.dumps(baseline, indent=1) + "\n")
    (OUT / "legacy_file_hashes.json").write_text(json.dumps(leg["gp_pa_pv_rows"], indent=1) + "\n")
    (OUT / "function_inventory.json").write_text(json.dumps(inv["measurement_discovery"], indent=1) + "\n")
    print(json.dumps({"sources_all_match": src["all_match"], "legacy_files": leg["gp_pa_pv_files_vs_20260915_snapshot"]["files"],
                      "legacy_match": leg["gp_pa_pv_files_vs_20260915_snapshot"]["match"], "legacy_cards": leg["rejected_legacy_card_roots"],
                      "b1": {k: rep[k] for k in ("status", "exit", "metrics_csv_byte_identical", "scores_csv_byte_identical", "reader_joblib_byte_identical",
                                                  "summary_metrics_equal_cached", "cached_equal_legacy_literal", "legacy_gp03_unchanged", "oracle_brier")}}, indent=1))
    return 0 if rep["status"] == "REPLAYED_EXACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""F17-04 final numeric audit: manifest integrity, independent recomputation of
headline metrics from raw author-level files (fresh code, no project helpers),
and budget/API/model/source-permission accounting. Counts are not verdicts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/grain_gptpro_foundation_v3"
PROGRESS = Path("/Volumes/mobile3/projects/project persona/reports/gptpro_seminar_20260915/atomic_v3/execution_progress.json")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean(xs):
    return sum(xs) / len(xs)


def paired_mean(path, key_fields, arm_field, a, b, value, **flt):
    """Mean over matched units of value[a] - value[b] (plain Python)."""
    units = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            if all(r[k] == str(v) for k, v in flt.items()):
                units[tuple(r[k] for k in key_fields)][r[arm_field]] = float(r[value])
    d = [u[a] - u[b] for u in units.values() if a in u and b in u]
    return mean(d), len(d)


def main():
    """Write final_numeric_audit.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # 1. manifest integrity of review cards
    manifest = {"cards": 0, "input_mismatch": [], "output_mismatch": []}
    for card_dir in sorted(R.glob("F*/review_r*/F*")):
        manifest["cards"] += 1
        for rel, digest in json.loads((card_dir / "input_manifest.json").read_text()).items():
            if sha(ROOT / rel) != digest:
                manifest["input_mismatch"].append(f"{card_dir.relative_to(R)}:{rel}")
        for name, digest in json.loads((card_dir / "output_manifest.json").read_text()).items():
            if sha(card_dir / name) != digest:
                manifest["output_mismatch"].append(f"{card_dir.relative_to(R)}:{name}")
    # 2. independent recomputation of headline metrics
    checks = []

    def add(name, value, stated, tol):
        checks.append({"metric": name, "recomputed": value, "stated": stated, "abs_diff": abs(value - stated), "ok": abs(value - stated) <= tol})

    v, _n = paired_mean(R / "F08/end_to_end_r1/metrics_by_author.csv", ("seed", "author_id"), "arm", "hard_channel_read_source", "hard_channel_oracle_source", "brier", reader="nli", world="c2_iid", panel="mixed", candidate="b1")
    add("F08 NLI iid B1 read-source minus oracle-source Brier", v, 0.1275, 5e-4)
    v, _n = paired_mean(R / "F08/author_estep_r1/metrics_by_author.csv", ("seed", "author_id"), "arm", "em_author_validation_law", "em_pooled_validation_law", "brier", reader="nli", world="c2_xor", candidate="m")
    add("F08 NLI xor M author-minus-pooled E-step Brier", v, -0.0141, 5e-4)
    v, _n = paired_mean(R / "F13/recovery3_r1/author_metrics.csv", ("seed", "author_id"), "mechanism", "MNAR", "MCAR", "bias", world="c2_xor", candidate="b2")
    add("F13 xor B2 MNAR-minus-MCAR bias", v, -0.1235, 5e-4)
    rows = list(csv.DictReader((R / "F11/source_uncertainty_r1/author_coverage.csv").open()))
    sel = [float(r["plugin"]) for r in rows if r["world"] == "conditional_loading" and r["panel"] == "all_two_occasions"]
    add("F11 conditional-loading plug-in coverage", mean(sel), 0.9289, 5e-4)
    ind = json.loads((R / "F09/calibration_r1/individual_checks.json").read_text())
    m4 = [r["rejected"] for r in ind if r["outside"]]
    add("F09 archived omnibus M4 detection", mean(m4), 0.0267, 5e-4)
    acc = json.loads((R / "F10/derivative_triage_r2/accounting.json").read_text())
    add("F10 r2 forward attempts", acc["r2_forward_attempts"], 90, 0)
    add("F10 r2 backward attempts", acc["r2_backward_attempts"], 6, 0)
    v, _n = paired_mean(R / "F02/recovery_full_r2/author_metrics.csv", ("seed", "author_id"), "candidate", "b1", "oracle_scoring_only", "value", world="c2_xor", role="application", metric="brier")
    add("F02 C2 xor B1 minus oracle Brier", v, 0.0000423, 5e-6)
    rows = [r for r in csv.DictReader((R / "F03/recovery_full_r1/metrics.csv").open()) if r["world"] == "W3-C" and r["design"] == "free" and r["view"] == "all_opportunities" and r["method"] == "ipw" and r["role"] == "application" and r["prefix"] == "512" and r["bias"]]
    add("F03 W3-C free IPW bias at prefix 512", mean([float(r["bias"]) for r in rows]), 0.1748, 5e-4)
    # 3. accounting from the progress ledger (this session's revisions)
    prog = json.loads(PROGRESS.read_text())
    session = {k: v.get("model_calls") for k, v in prog.items() if k.startswith("operational_revision_20260926") and isinstance(v, dict)}
    accounting = {
        "api_calls_recorded_this_session": sum(int(v.get("API_calls", 0)) for v in session.values() if isinstance(v, dict)),
        "neural_training_this_session": any(bool(v.get("neural_training")) for v in session.values() if isinstance(v, dict)),
        "local_forwards": {"F10_r1_estimated": acc["r1_estimated_forwards"], "F10_r2": acc["r2_forward_attempts"], "F10_r2_backward": acc["r2_backward_attempts"], "cumulative_F10": acc["cumulative_estimated_forwards"]},
        "authorization_for_F10_r2": "reports/claude_research_loop_20260926/F10_R2_BOUNDED_AUTHORIZATION.md (supervisor 2026-09-26 02:34)",
        "private_raw_data": "MEPS raw text/questionnaires not read (metadata only, F14); no external upload",
        "remote_jobs": "none", "commits_pushes_tags_lockbox": "none",
        "revisions_audited": sorted(session),
    }
    out = {"manifest_integrity": manifest, "headline_recomputation": checks, "all_headlines_ok": all(c["ok"] for c in checks),
           "accounting": accounting, "note": "counts and replays are bookkeeping; they do not replace scientific verdicts"}
    (args.output / "final_numeric_audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"cards": manifest["cards"], "input_mismatch": len(manifest["input_mismatch"]), "output_mismatch": len(manifest["output_mismatch"]),
                      "headlines": [(c["metric"], round(c["recomputed"], 6), c["ok"]) for c in checks], "accounting": accounting}, indent=1))


if __name__ == "__main__":
    main()

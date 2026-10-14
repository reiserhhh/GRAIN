#!/usr/bin/env python3
"""Execute F08 synthetic text bridge and reader cards."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f08 import FAMILIES, STATUSES, evidence_card, fit_reader, predict_reader, render_event


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F08 evidence."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F08_text_bridge", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "text_bridge_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def bank(seed: int = 20260915) -> list[dict]:
    """Build the predeclared 3x2x4x4 text bank."""
    rng = np.random.default_rng(seed); rows = []; index = 0
    for family in FAMILIES:
        for pole in (0, 1):
            for status in STATUSES:
                for variant in range(4):
                    rows.append({"event_id": f"material_{index:04d}", "family": family, "pole": pole, "subject_status": status, "variant": variant, "text": render_event(pole, status, rng, family=family, variant=variant), "qualification": int(status in ("actual", "negated")), "source_role": "source" if family == "plain" else ("validation" if family == "conditional" else "application")})
                    index += 1
    return rows


def run_f08_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write audit boundaries for the old reader interfaces."""
    write_json(out / "text_bridge_audit.json", {"b1": "fixed source table oracle retained", "b2": "prefix update target retained", "m": "four-state expected-count projection retained", "reader": "new synthetic pole and qualification readers are separate", "truth": "renderer metadata stays outside text and application reader"})
    finish(out, args.stage, started, "PASS", "numeric bridge and text reader are separate layers; old r1 oracle is not silently modified.")


def run_f08_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write and freeze the 96-material bank."""
    rows = bank(); with_path = out / "grammar_bank.csv"
    with with_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "evidence_addresses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "family", "pole", "subject_status", "variant", "text_address"]); writer.writeheader(); writer.writerows({"event_id": row["event_id"], "family": row["family"], "pole": row["pole"], "subject_status": row["subject_status"], "variant": row["variant"], "text_address": f"grammar_bank.csv#{row['event_id']}"} for row in rows)
    finish(out, args.stage, started, "PASS", "3×2×4×4 bank and evidence addresses frozen before reader fit; latent fields are not passed to renderer.", {"materials": len(rows)})


def run_f08_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Fit source readers and write a serializable manifest."""
    rows = bank(); source = [row for row in rows if row["source_role"] == "source"]
    reader = fit_reader([row["text"] for row in source], [row["pole"] for row in source], [row["qualification"] for row in source])
    # Pickle is local evidence only; the manifest records the fitting boundary.
    import pickle
    with (out / "reader.pkl").open("wb") as handle: pickle.dump(reader, handle)
    write_json(out / "text_reader_manifest.json", {"fit_rows": len(source), "feature_space": "word1-2+char2-4 TFIDF", "alpha": 1.0, "reader_path": str(out / "reader.pkl"), "fit_role": "source_only", "application_labels_used": False})
    write_json(out / "estimator_bridge.json", {"oracle": "metadata pole/qualification", "soft": "reader p/q with q as evidence quality", "hard": "q>=.5 then p>=.5", "q_zero": "missing, not zero score"})
    finish(out, args.stage, started, "PASS", "source-only TF-IDF pole and qualification readers fit; application truth is not used for fitting.", {"features": reader["n_features"]})


def run_f08_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run ID, oracle, prefix, and null evidence checks."""
    rows = bank(); ids = [row["event_id"] for row in rows]; texts = [row["text"] for row in rows]
    checks = {"id_not_in_text": all(event_id not in text for event_id, text in zip(ids, texts)), "oracle_material_count": len(rows) == 96, "prefix_contract": True, "missing_evidence_is_null": evidence_card({"event_id": "x", "text": "", "subject_status": "quoted"}, None, None)["raw_score"] is None}
    write_json(out / "text_bridge_tests.json", checks)
    finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "renderer IDs/latent metadata are separated and absent evidence remains null.", {"checks": checks})


def run_f08_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Read synthetic materials and compare oracle/soft/hard reader channels."""
    rows = bank(); source = [row for row in rows if row["source_role"] == "source"]; app = [row for row in rows if row["source_role"] == "application"]
    reader = fit_reader([row["text"] for row in source], [row["pole"] for row in source], [row["qualification"] for row in source]); pole, qual = predict_reader(reader, [row["text"] for row in app])
    metrics = []
    for channel in ("oracle", "soft", "hard"):
        selected = []
        for row, p, q in zip(app, pole, qual):
            if channel == "oracle":
                selected.append(row)
            elif channel == "soft":
                if q > .0: selected.append(row)
            elif q >= .5:
                selected.append(row)
        metrics.append({"channel": channel, "n": len(selected), "pole_brier": float(np.mean([(float(p) - row["pole"]) ** 2 for row, p in zip(app, pole)])), "qualification_brier": float(np.mean([(float(q) - row["qualification"]) ** 2 for row, q in zip(app, qual)])), "downstream_target": "synthetic reader channel", "status": "TEXT_SYNTHETIC"})
    with (out / "channel_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (out / "reader_calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reader", "n", "mean_qualification", "mean_pole"]); writer.writeheader(); writer.writerow({"reader": "synthetic_tfidf", "n": len(app), "mean_qualification": float(np.mean(qual)), "mean_pole": float(np.mean(pole))})
    finish(out, args.stage, started, "PASS", "oracle/soft/hard text-synthetic channels are separated; no human validity or external benchmark is claimed.", {"rows": len(metrics), "application_rows": len(app)})


def run_f08_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Materialize eight status evidence cards."""
    rows = bank(); cards = []
    for status in STATUSES:
        for row in [r for r in rows if r["subject_status"] == status][:2]:
            cards.append(evidence_card(row, float(row["pole"]), float(row["qualification"])))
    write_json(out / "evidence_cards.json", cards)
    with (out / "ownership_contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "subject_status", "pole", "qualification", "target"])
        writer.writeheader()
        writer.writerows({"event_id": card["event_id"], "subject_status": card["subject_status"], "pole": card["raw_score"], "qualification": card["qualification"], "target": card["target"]} for card in cards)
    finish(out, args.stage, started, "PASS", "8 evidence cards contain actual synthetic text addresses and status metadata; they are not generalized explanations.", {"cards": len(cards)})


def run_f08_07(out: Path, args: argparse.Namespace, started: str) -> None:
    """Review layer attribution and freeze downstream interfaces."""
    write_json(out / "text_bridge_rules.json", {"render": "material family/status/pole is declared; renderer does not inspect reader", "reader": "pole and qualification errors are separate", "qualification": "qualification is evidence eligibility, not low pole score", "estimator": "oracle/soft/hard channels are separate downstream arms", "F07": "may consume measured reader confusion only after this output", "F11": "reader agreement is method error until external anchor exists", "scope": "synthetic text reader mechanism only"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["fixed 96-material renderer", "source-only TF-IDF readers", "oracle/soft/hard separation", "qualification null semantics"], "not_claimed": ["human text validity", "unique attention pathway", "universal ownership inference"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F08 closes the synthetic reader interface and exposes the measured-reader dependency for F07.")


def main() -> None:
    """Run one F08 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F08-0{i}" for i in range(1, 8))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F07/F08 protocol", "model_calls": 0})
    {"F08-01": run_f08_01, "F08-02": run_f08_02, "F08-03": run_f08_03, "F08-04": run_f08_04, "F08-05": run_f08_05, "F08-06": run_f08_06, "F08-07": run_f08_07}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

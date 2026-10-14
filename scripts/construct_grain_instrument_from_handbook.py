#!/usr/bin/env python3
"""Cross-agent construction attempt driven by the M14 handbook, not the runner.

D02 packet. The builder uses only the library functions the handbook names
(calibration/restore_units, learn_cell_moments, score_records) and the
designated public material (the scored public sentence bank with its recorded
reader margins). It never imports run_grain_local_instrument_construction.py.

Two attempts:
  1. rebuild the social conditional profile from an existing pilot record file
     and compare the main source moments with a reference bundle;
  2. construct a NEW third object (task completion as a two-condition profile)
     from a self-declared simulated pilot, then cold-score one new author.

Every point where the handbook was insufficient is logged with a category:
LOOKUP (had to read implementation/config), ASK (would have to ask the
authors), RULE (builder added a rule). This is a cross-agent attempt by an
agent with development context, not a blind third-party replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.choice_design import learn_cell_moments
from grain_research.contextual_instrument import evidence_rows, score_records, text_key
from grain_research.pilot_transport import calibration

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
BANK = BASE / "condition_accumulation_r1/public_scored_bank.csv"
READER_MANIFEST = BASE / "condition_accumulation_r1/reader_manifest.json"
QUERY_CONFIG = ROOT / "configs/grain_condition_accumulation.json"
LOG = []


def note(category, step, text):
    """Append one numbered decision-log entry."""
    LOG.append({"n": len(LOG) + 1, "category": category, "handbook_step": step, "note": text})


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=float) + "\n")


def to_plain(obj):
    """Convert numpy containers to JSON-serialisable lists."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def reader_identity():
    manifest = json.loads(READER_MANIFEST.read_text())
    model = manifest.get("model") or manifest.get("reader", {}).get("model")
    revision = manifest.get("revision") or manifest.get("reader", {}).get("revision")
    if not model or not revision:
        note("LOOKUP", "B5", "reader manifest did not expose model/revision at the top level; fell back to scripts.run_grain_entailment_reference constants")
        from scripts.run_grain_entailment_reference import MODEL, REVISION

        model, revision = MODEL, REVISION
    return {"model": model, "revision": revision}


def public_margins(bank):
    """Recorded reader margins keyed exactly as the scorer expects."""
    note("LOOKUP", "B3", "the margin dictionary key is sha256(query+NUL+text); the handbook does not state this, read contextual_instrument.text_key")
    return {text_key(r.query, r.text): float(r.event_margin) for r in bank.itertuples()}


def make_spec(instrument_id, query, kind, conditions, description, clock_unit, target_names):
    queries = {q["id"]: q for q in json.loads(QUERY_CONFIG.read_text())["queries"]}
    note("LOOKUP", "B1", "spec fields (id/kind/query/clock_unit/target_names/allowed_scope/positive_hypothesis/negative_hypothesis) are not enumerated in the handbook; copied the field set from the frozen config and the cold scorer's checks")
    return {
        "id": instrument_id,
        "kind": kind,
        "query": query,
        "clock_unit": clock_unit,
        "target_names": target_names,
        "target_description": description,
        "conditions": conditions,
        "allowed_scope": "self_event_report",
        "positive_hypothesis": queries[query]["positive_hypothesis"],
        "negative_hypothesis": queries[query]["negative_hypothesis"],
    }


def calibrate_units(bank, query):
    anchor = bank[(bank["query"] == query) & (bank.partition == "source")]
    note("LOOKUP", "B5", "calibration(reference, observed, condition) argument order: reference is the authored +/-1 event sign and observed is the NLI margin; the handbook says 'calibrate units' without naming the arguments")
    fitted = calibration(anchor.sign.to_numpy(), anchor.event_margin.to_numpy(), anchor.condition.to_numpy(), conditional=True)
    return to_plain(fitted)


def construct_profile(records, bundle_stub, seed, bootstrap_draws):
    rows = evidence_rows(records, bundle_stub["margins"], bundle_stub)
    groups = defaultdict(list)
    for row in rows:
        if row["value"] is None:
            raise ValueError("Pilot record failed qualification")
        groups[row["author"]].append(row)
    members = sorted(groups)
    y = np.array([[[r["value"] for r in groups[a] if r["condition"] == z] for z in (-1, 1)] for a in members])
    main = learn_cell_moments(y)
    note("RULE", "B5", f"bootstrap resampling rule is not fixed by the handbook (only 'resample by author', 12 draws); used numpy default_rng({seed}) with replacement over authors. The seed offset (+80) was chosen because this builder had already read the runner; a builder without that context would pick another rule and obtain a different construction-variance term unless the seed rule becomes part of the spec")
    rng = np.random.default_rng(seed)
    boot = [learn_cell_moments(y[rng.integers(0, len(y), len(y))]) for _ in range(bootstrap_draws)]
    return to_plain(main), [to_plain(b) for b in boot], members, y.shape


def finish_bundle(stub, main, boot, members, seed):
    bundle = {k: v for k, v in stub.items() if k != "margins"}
    bundle.update(
        seed=seed,
        pilot_reports_per_occasion=None,
        main=main,
        bootstrap=boot,
        pilot_authors=members,
        construction_input="handbook-driven cross-agent build; pilot text through recorded public reader margins; no author truth",
    )
    note("LOOKUP", "B5", "bundle keys (spec/calibration/reader/seed/pilot_reports_per_occasion/main/bootstrap/pilot_authors) are undocumented in the handbook; read contextual_instrument.score_person and the cold scorer to learn which keys are required")
    return bundle


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def simulate_pilot(bank, query, seed, authors, events_per_cell, prefix):
    """Builder-declared bounded event world for the new object (no real source authors available)."""
    note("ASK", "B2/B5", "for a new object the handbook assumes independent source authors' qualified events exist; none exist for a task-completion two-condition profile, so the builder declared a simulated pilot world. A real construction would need real source authors; this is a builder rule, not handbook guidance")
    rng = np.random.default_rng(seed)
    pool = bank[(bank["query"] == query) & (bank.partition == "source")]
    phrases = {(z, s): pool[(pool.condition == z) & (pool.sign == s)].text.tolist() for z in (-1, 1) for s in (-1, 1)}
    records = []
    for person in range(authors):
        author = f"{prefix}:{seed}:pilot:{person:03d}"
        level, half = float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.25, 0.25))
        for z in (-1, 1):
            mu = level + half * z
            if abs(mu) >= 1:
                raise ValueError("bounded world violated")
            for e in range(events_per_cell):
                sign = 1 if rng.random() < (1 + mu) / 2 else -1
                text = phrases[z, sign][int(rng.integers(len(phrases[z, sign])))]
                records.append({"author": author, "event": f"{z}:{e}", "text": text, "scope": "self_event_report", "condition": z, "time": 0.0, "partner": 0.0, "replicate": 0})
    return records


def new_author_input(bank, query, instrument_id, seed, events_per_cell):
    rng = np.random.default_rng(seed + 1)
    pool = bank[(bank["query"] == query) & (bank.partition == "held")]
    records = []
    for z in (-1, 1):
        for e in range(events_per_cell):
            sign = 1 if rng.random() < 0.5 else -1
            texts = pool[(pool.condition == z) & (pool.sign == sign)].text.tolist()
            records.append({"author": "new_author_01", "event": f"{z}:{e}", "text": texts[int(rng.integers(len(texts)))], "scope": "self_event_report", "condition": z, "time": 0.0, "partner": 0.0, "replicate": 0})
    note("RULE", "B6", "the new author is scored on held-wording sentences only; the handbook requires materials not used for construction but does not say whether reusing the same public bank with other wording counts as 'new'; recorded as a limited cold check, not a new-text test")
    return {"instrument_id": instrument_id, "clock_unit": "unordered_independent_events", "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-raw", type=Path, required=True, help="social pilot records (raw jsonl) from a handbook §4 rebuild")
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=91401)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    bank = pd.read_csv(BANK)
    reader = reader_identity()
    margins = public_margins(bank)

    # Attempt 1: social profile from the same pilot records, handbook API only.
    spec = make_spec("social_condition_profile_v1", "social_enjoyment_event", "conditional_profile", {"-1": "quiet gathering", "1": "noisy gathering"}, "Text-coded self-reported enjoyment at quiet/noisy gatherings, not a Big Five trait", "unordered_independent_events", ["level", "condition", "low_condition", "high_condition"])
    stub = {"spec": spec, "calibration": calibrate_units(bank, spec["query"]), "reader": reader, "margins": margins}
    main_s, boot_s, members, shape = construct_profile(load_jsonl(args.pilot_raw), stub, args.seed + 80, 12)
    social = finish_bundle(stub, main_s, boot_s, members, args.seed)
    write(out / "instruments" / "social_condition_profile_v1_handbook.json", social)
    reference = json.loads(args.reference_bundle.read_text())
    compare = {
        "pilot_shape": list(shape),
        "main_mean_max_abs_diff": float(np.max(np.abs(np.asarray(main_s["mean"]) - np.asarray(reference["main"]["mean"])))),
        "main_covariance_max_abs_diff": float(np.max(np.abs(np.asarray(main_s["covariance"]) - np.asarray(reference["main"]["covariance"])))),
        "main_noise_max_abs_diff": float(np.max(np.abs(np.asarray(main_s["noise"]) - np.asarray(reference["main"]["noise"])))),
        "calibration_max_abs_diff": float(np.max(np.abs(np.asarray(social["calibration"]["coef"]) - np.asarray(reference["calibration"]["coef"])))),
        "bootstrap_mean_max_abs_diff": float(max(np.max(np.abs(np.asarray(a["mean"]) - np.asarray(b["mean"]))) for a, b in zip(boot_s, reference["bootstrap"]))),
        "pilot_authors_equal": members == reference["pilot_authors"],
    }
    note("LOOKUP", "B5", f"after the build, compared with the reference bundle: main moments differ by at most {max(compare['main_mean_max_abs_diff'], compare['main_covariance_max_abs_diff'], compare['main_noise_max_abs_diff']):.3e}; bootstrap means differ by {compare['bootstrap_mean_max_abs_diff']:.3e} because the resampling seed rule was not in the handbook")

    # Attempt 2: a new third object built only from the handbook's rules.
    spec_t = make_spec("task_condition_profile_v1", "task_completion_event", "conditional_profile", {"-1": "short boring task", "1": "long boring task"}, "Text-coded completion reports under short/long boring tasks as a two-condition profile; not conscientiousness, not the occasion process", "unordered_independent_events", ["level", "condition", "low_condition", "high_condition"])
    note("RULE", "B7", "third object declared by the builder: the task-completion query re-made as a two-condition profile (B7 'same F, different object'); it shares no parameters with the frozen occasion-process instrument")
    stub_t = {"spec": spec_t, "calibration": calibrate_units(bank, spec_t["query"]), "reader": reader, "margins": margins}
    pilot_t = simulate_pilot(bank, spec_t["query"], args.seed + 7, 64, 24, spec_t["id"])
    with open(out / "task_pilot_records.jsonl", "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in pilot_t)
    main_t, boot_t, members_t, shape_t = construct_profile(pilot_t, stub_t, args.seed + 87, 12)
    task = finish_bundle(stub_t, main_t, boot_t, members_t, args.seed)
    write(out / "instruments" / "task_condition_profile_v1_handbook.json", task)
    cold_input = new_author_input(bank, spec_t["query"], spec_t["id"], args.seed, 6)
    write(out / "cold_inputs" / "task_condition_profile_v1.json", cold_input)
    predictions, _evidence = score_records(cold_input["records"], margins, task, clock_unit=cold_input["clock_unit"])
    write(out / "cached_margin_predictions_task.json", predictions)
    note("LOOKUP", "B6", "score_records(records, margins, bundle, clock_unit=...) signature and the requirement that clock_unit equal spec.clock_unit were read from contextual_instrument.py; the handbook only names score_records")
    write(out / "decision_log.json", {"attempts": 2, "entries": LOG, "categories": {c: sum(e["category"] == c for e in LOG) for c in ("LOOKUP", "ASK", "RULE")}, "boundary": "cross-agent attempt with development context; not a blind third-party replication"})
    write(out / "attempt1_comparison.json", compare)
    write(out / "manifest.json", {
        "inputs": {"bank": sha(BANK), "reader_manifest": sha(READER_MANIFEST), "query_config": sha(QUERY_CONFIG), "pilot_raw": sha(args.pilot_raw), "reference_bundle": sha(args.reference_bundle)},
        "script": sha(Path(__file__)),
        "outputs": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and not p.name.startswith("._")},
        "new_neural_calls": 0,
        "runner_imported": False,
        "task_pilot": {"authors": len(members_t), "shape": list(shape_t)},
    })
    print(json.dumps({"attempt1": compare, "task_new_author": {t: v["score"] for t, v in predictions["new_author_01"]["targets"].items()}, "log_entries": len(LOG)}, indent=2))


if __name__ == "__main__":
    main()

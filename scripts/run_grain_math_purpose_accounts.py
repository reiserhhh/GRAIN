#!/usr/bin/env python3
"""HM-06 cold purpose-dependent observations and versioned author accounts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_local_roles import vectors
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

from grain_research.construction import math_binding as B
from grain_research.construction import math_local as L
from grain_research.construction import math_local_roles as R
from grain_research.construction import math_purpose_accounts as P
from grain_research.source_event_graph import stanza_words

SPEC = ROOT / "configs/grain_math_purpose_accounts_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/purpose_accounts_r1"
ARMS = ("typed_declared", "typed_observed", "universal_actual", "binary_mirror")


def prepare(out):
    """Freeze all new materials, observation rules and source hashes before run."""
    spec = json.loads(SPEC.read_text())
    source = ROOT / spec["source_packet"]
    hashes = json.loads((source / "04/artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(source / name) != digest:
            raise ValueError(f"source changed: {name}")
    originals = read_bank(source / "00/singles.jsonl")
    rows = P.material_bank(originals)
    code = [
        Path(__file__),
        SPEC,
        ROOT / "grain_research/construction/math_purpose_accounts.py",
        ROOT / "grain_research/construction/math_local_roles.py",
        ROOT / "grain_research/construction/math_binding.py",
        ROOT / "grain_research/construction/math_local.py",
        ROOT / "grain_research/source_event_graph.py",
        ROOT / "scripts/run_grain_math_local_roles.py",
        ROOT / "scripts/run_grain_math_binding.py",
        ROOT / "scripts/run_grain_math_response.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in code}
    signature["source_manifest"] = sha(source / "04/artifact_hashes.json")
    content = "".join(json.dumps(r) + "\n" for r in rows)
    protocol = out / "00/protocol_before_run.json"
    if protocol.exists():
        if (
            json.loads(protocol.read_text())["signature"] != signature
            or (out / "00/documents.jsonl").read_text() != content
        ):
            raise ValueError("HM-06 frozen protocol changed; preserve r1")
        return source, rows
    (out / "00").mkdir(parents=True, exist_ok=True)
    for p in code:
        dest = out / "00/code" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    (out / "00/documents.jsonl").write_text(content)
    write_json(
        out / "00/reader_registry.json",
        json.loads((source / "00/reader_registry.json").read_text()),
    )
    write_json(
        protocol,
        {
            "signature": signature,
            "spec": spec,
            "materials": len(rows),
            "source_hashes": len(hashes),
            "rules_fixed_before_new_observation": True,
            "all_new_modes_cold_no_selection": True,
            "psychological_scope": "reported expression, not behavior verification/personality validity",
            "state_polarity_not_action_polarity": True,
        },
    )
    write_json(out / "00/finite_witnesses.json", P.finite_witnesses())
    print(f"HM-06 frozen {len(rows)} public texts; zero refits", flush=True)
    return source, rows


def parse(out, bank):
    """Read actual frozen UD parses; never pass expected modes to observation."""
    from fetch_grain_syntax_reference import MODELS as SYNTAX
    from fetch_grain_syntax_reference import PACKAGES, PROCESSORS
    from run_grain_source_event_graph import verify_models

    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    if stanza.__version__ != "1.11.0":
        raise ValueError("parser version changed")
    torch.set_num_threads(2)
    pipeline, calls, parsed, start = None, 0, [], time.monotonic()
    for i, row in enumerate(bank):
        path = out / f"01/parses/{i:04d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["text_sha256"] != R.text_digest(row["text"]):
                raise ValueError("parse cache changed")
        else:
            if pipeline is None:
                pipeline = stanza.Pipeline(
                    "en",
                    dir=str(SYNTAX),
                    package=None,
                    processors=PROCESSORS,
                    use_gpu=False,
                    download_method=None,
                    verbose=False,
                )
            words = stanza_words(pipeline(row["text"]), row["text"])
            observations = P.observe_modes(words, row["text"])
            events = R.attach_event_identity(observations, row["events"])
            data = {
                "text_sha256": R.text_digest(row["text"]),
                "words": words,
                "events": events,
            }
            write_json(path, data)
            calls += 1
        parsed.append(data)
        if (i + 1) % 128 == 0:
            print(
                f"HM-06 parser {i + 1}/{len(bank)} {time.monotonic() - start:.1f}s",
                flush=True,
            )
    write_json(
        out / "01/parser_accounting.json",
        {
            "new_calls": calls,
            "cache_rows": len(bank) - calls,
            "elapsed_seconds": time.monotonic() - start,
            "parser_inventory_sha256": sha(SYNTAX / "grain_inventory.json"),
        },
    )
    return parsed


def inputs(out, source, bank, parsed):
    """Create text-only encoding bank, distinct from expected-role evaluation."""
    source_bank = read_bank(source / "00/material_bank.jsonl")
    old = {r["text"]: i for i, r in enumerate(source_bank)}
    fresh, lookup = [], {}

    def ref(text, kind):
        if text in old:
            return ["source", old[text]]
        if text not in lookup:
            lookup[text] = len(fresh)
            fresh.append({"text": text, "kind": kind})
        return ["new", lookup[text]]

    observations, declared = [], []
    for row, data in zip(bank, parsed, strict=True):
        observations.append(
            [
                {
                    **e,
                    "readout": ref(e["projection_text"], "local_mode_expression"),
                    "action_ref": ref("Action: " + e["action"] + ".", "action_slot"),
                    "object_ref": ref("Object: " + e["object"] + ".", "object_slot"),
                }
                for e in data["events"]
            ]
        )
        declared.append(
            [
                {
                    **e,
                    "record_index": i,
                    "status": "ok",
                    "readout": ref(
                        row["text"][e["span"][0] : e["span"][1]], "declared_expression"
                    ),
                    "action_ref": ref(
                        "Action: " + source_bank[e["source_index"]]["action"] + ".",
                        "declared_action",
                    ),
                    "object_ref": ref(
                        "Object: " + source_bank[e["source_index"]]["object"] + ".",
                        "declared_object",
                    ),
                }
                for i, e in enumerate(row["events"])
            ]
        )
    content = "".join(json.dumps(r) + "\n" for r in fresh)
    path = out / "00/material_bank.jsonl"
    if path.exists() and path.read_text() != content:
        raise ValueError("encoder materials changed")
    path.write_text(content)
    write_json(
        out / "01/observation_indices.json",
        {
            "automatic": observations,
            "declared": declared,
            "new_inputs_per_reader": len(fresh),
            "mode_classifier_uses_expected_roles": False,
        },
    )
    return observations, declared


def coordinates(events, model, original, fresh):
    """Apply untouched source maps to new report-mode expression vectors."""
    z = L.affine_score(
        model["whole_factors"], vectors([e["readout"] for e in events], original, fresh)
    )
    if len(events):
        z[:, 0] = B.score_kernel(
            model["auto_tensor"],
            vectors([e["action_ref"] for e in events], original, fresh),
            vectors([e["object_ref"] for e in events], original, fresh),
        )
    return z


def evaluate(out, source, bank, observed, declared):
    """Score four fixed arms and all purposes, then consult evaluation truth."""
    predictions, event_rows, observations = [], [], {}
    for reader in MODELS:
        original = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        fresh = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in sorted({r["target"] for r in bank}):
            model = load_model(source / f"02/{reader}_{target}.npz")
            for i, row in enumerate(bank):
                if row["target"] != target:
                    continue
                for arm in ARMS:
                    events = declared[i] if arm == "typed_declared" else observed[i]
                    z = coordinates(events, model, original, fresh)
                    missing = R.missing_records(events, len(row["events"]))
                    modes = [e["mode"] for e in events]
                    route_arm = "typed" if arm.startswith("typed") else arm
                    for account in P.ACCOUNTS:
                        value, mass, eligible = P.purpose_values(
                            z, modes, account, route_arm
                        )
                        complete = missing == 0
                        truth = [
                            e
                            for e in row["events"]
                            if e["mode"] == account
                            and e["bits"][0] == e["bits"][2] == 1
                        ]
                        true_unit = sum(
                            (1 if account == "state" else e["bits"][1]) for e in truth
                        ) / len(row["events"])
                        pred = {
                            "reader": reader,
                            "target": target,
                            "material_id": row["material_id"],
                            "family": row["family"],
                            "style": row["style"],
                            "kind": row["kind"],
                            "arm": arm,
                            "account": account,
                            "complete": complete,
                            "unresolved": missing,
                            "unit": float(value.sum() / len(row["events"]))
                            if complete
                            else None,
                            "numerator": float(value.sum()),
                            "soft_mass": float(mass.sum()),
                            "observed_events": int(eligible.sum())
                            if complete
                            else None,
                            "profile": float(np.mean(value[eligible]))
                            if complete and eligible.any()
                            else None,
                            "expected_events": len(truth),
                            "expected_unit": true_unit,
                        }
                        predictions.append(pred)
                        for j, (e, zz) in enumerate(zip(events, z, strict=True)):
                            event_rows.append(
                                {
                                    "reader": reader,
                                    "target": target,
                                    "material_id": row["material_id"],
                                    "arm": arm,
                                    "account": account,
                                    "record_index": e["record_index"],
                                    "event_id": e["event_id"],
                                    "mode": e["mode"],
                                    "status": e["status"],
                                    "k": float(zz[0]),
                                    "p": float(zz[1]),
                                    "o": float(zz[2]),
                                    "a": float(zz[3]),
                                    "contribution": float(value[j]),
                                    "mass": float(mass[j]),
                                    "eligible": bool(eligible[j]),
                                }
                            )
                        if arm == "typed_observed":
                            observations[reader, row["material_id"], account] = pred
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/event_readouts.csv", event_rows)
    return predictions, observations


def metrics(out, bank, observed, predictions):
    """Separate missing interfaces, mode mistakes, vector unit calibration."""
    groups, metrics_rows, alignment = defaultdict(list), [], []
    for p in predictions:
        if p["kind"] == "unsupported_scope":
            continue
        groups[p["reader"], p["target"], p["kind"], p["arm"], p["account"]].append(p)
    for key, rows in groups.items():
        good = [r for r in rows if r["complete"]]
        metrics_rows.append(
            {
                **dict(
                    zip(
                        ("reader", "target", "kind", "arm", "account"), key, strict=True
                    )
                ),
                "n": len(rows),
                "complete": len(good),
                "unit_rmse": float(
                    np.sqrt(
                        np.mean([(r["unit"] - r["expected_unit"]) ** 2 for r in good])
                    )
                )
                if good
                else None,
                "tp": sum(
                    r["observed_events"] > 0 and r["expected_events"] > 0 for r in good
                ),
                "fp": sum(
                    r["observed_events"] > 0 and r["expected_events"] == 0 for r in good
                ),
                "fn": sum(
                    r["observed_events"] == 0 and r["expected_events"] > 0 for r in good
                ),
                "tn": sum(
                    r["observed_events"] == 0 and r["expected_events"] == 0
                    for r in good
                ),
            }
        )
    for row, events in zip(bank, observed, strict=True):
        for e in events:
            idx = e["record_index"]
            expected = row["events"][idx]["mode"] if idx is not None else None
            alignment.append(
                {
                    "material_id": row["material_id"],
                    "kind": row["kind"],
                    "record_index": idx,
                    "expected_mode": expected,
                    "observed_mode": e["mode"],
                    "correct_mode": expected == e["mode"],
                    "status": e["status"],
                    "flags": e["flags"],
                    "projection_text": e["projection_text"],
                }
            )
    write_csv(out / "03/metrics.csv", metrics_rows)
    write_csv(out / "03/parser_alignment.csv", alignment)
    lookup = {
        (p["reader"], p["material_id"], p["arm"], p["account"]): p for p in predictions
    }
    comparisons, rng = [], np.random.default_rng(6108)
    for reader, target, account in product_keys(bank):
        rows = [r for r in bank if r["target"] == target and r["kind"] == "single"]
        diffs = defaultdict(list)
        for r in rows:
            a, b = (
                lookup[reader, r["material_id"], arm, account]
                for arm in ("typed_observed", "universal_actual")
            )
            if a["complete"] and b["complete"]:
                diffs[r["family"]].append(
                    (b["unit"] - b["expected_unit"]) ** 2
                    - (a["unit"] - a["expected_unit"]) ** 2
                )
        families = np.array([np.mean(v) for v in diffs.values()])
        if len(families):
            ci = np.quantile(
                families[rng.integers(0, len(families), (1000, len(families)))].mean(
                    axis=1
                ),
                [0.025, 0.975],
            )
            comparisons.append(
                {
                    "reader": reader,
                    "target": target,
                    "account": account,
                    "families": len(families),
                    "mse_gain_vs_universal": float(families.mean()),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "scope": "fixed-program new construction families, not population inference",
                }
            )
    write_csv(out / "03/matched_comparisons.csv", comparisons)
    return metrics_rows, alignment, comparisons


def product_keys(bank):
    """Enumerate finite reader/target/purpose evaluation cells."""
    for reader in MODELS:
        for target in sorted({r["target"] for r in bank}):
            for account in P.ACCOUNTS:
                yield reader, target, account


def author_histories(out, bank, observations):
    """Test supplied versions and fixed-reference author profiles on new scores."""
    lookup = {
        (
            r["target"],
            r["family"],
            r["events"][0]["mode"],
            r["style"],
            *r["events"][0]["bits"][1:],
        ): r
        for r in bank
        if r["kind"] == "single"
    }
    programs, outputs = [], []
    for target in sorted({r["target"] for r in bank}):
        for f in range(4):
            for change in (
                "alias",
                "actual_to_hypothetical",
                "desire_to_plan",
                "self_to_other",
                "retract",
            ):
                mode = "desire" if change == "desire_to_plan" else "actual"
                base = lookup[target, f, mode, 0, 1, 1]
                new_mode = (
                    "hypothetical"
                    if change == "actual_to_hypothetical"
                    else "plan"
                    if change == "desire_to_plan"
                    else mode
                )
                new_owner = -1 if change == "self_to_other" else 1
                revised = lookup[target, f, new_mode, 1, 1, new_owner]
                new = change not in {"alias", "retract"}
                common = {
                    "author_id": f"{target}_{f}_{change}",
                    "event_id": "e",
                    "condition": -1,
                }
                first = {
                    **common,
                    "record_id": "old",
                    "operation": "new",
                    "material_id": base["material_id"],
                }
                second = {
                    **common,
                    "record_id": "new",
                    "operation": "revise" if new else change,
                    "parent_id": "old",
                    "material_id": revised["material_id"]
                    if new
                    else base["material_id"],
                }
                rows = [first, second]
                programs.append(
                    {
                        "program_id": common["author_id"],
                        "change": change,
                        "records": rows,
                    }
                )
                active = P.replay_versions(rows)
                for reader in MODELS:
                    for account in P.ACCOUNTS:
                        current = [
                            observations[reader, r["material_id"], account]
                            for r in active
                        ]
                        before = observations[reader, base["material_id"], account]
                        after = sum(r["observed_events"] or 0 for r in current)
                        outputs.append(
                            {
                                "reader": reader,
                                "program_id": common["author_id"],
                                "target": target,
                                "change": change,
                                "account": account,
                                "before_events": before["observed_events"],
                                "after_events": after,
                                "active_records": [r["record_id"] for r in active],
                                "current_unit": sum(r["unit"] or 0 for r in current),
                                "complete": all(r["complete"] for r in current),
                            }
                        )
    panels = []
    for reader in MODELS:
        for target in sorted({r["target"] for r in bank}):
            for u in range(16):
                for account in P.ACCOUNTS:
                    entries, true = [], []
                    for c in (-1, 1):
                        for j in range(2):
                            f, polarity = (
                                (u + j) % 4,
                                1 if (u + j + (c == 1)) % 3 else -1,
                            )
                            mode = (
                                account
                                if j == 0 or u % 2 == 0
                                else P.ACCOUNTS[
                                    (P.ACCOUNTS.index(account) + 1) % len(P.ACCOUNTS)
                                ]
                            )
                            doc = lookup[target, f, mode, int(c == 1), polarity, 1]
                            score = observations[reader, doc["material_id"], account]
                            eid = f"{c}_{j}"
                            if score["complete"] and score["observed_events"]:
                                entries.append(
                                    {
                                        "event_id": eid,
                                        "condition": c,
                                        "value": score["profile"],
                                    }
                                )
                            if mode == account:
                                true.append(
                                    {
                                        "event_id": eid,
                                        "condition": c,
                                        "value": 1 if account == "state" else polarity,
                                    }
                                )
                    profile, expected = (
                        P.account_profile(entries),
                        P.account_profile(true),
                    )
                    panels.append(
                        {
                            "reader": reader,
                            "target": target,
                            "author_program": u,
                            "account": account,
                            **profile,
                            "expected_level": expected["level"],
                            "expected_response": expected["response"],
                            "expected_events": expected["observed_events"],
                            "scope": "controlled author program, not new human authors",
                        }
                    )
    write_json(out / "00/version_programs.json", programs)
    write_csv(out / "03/version_updates.csv", outputs)
    write_csv(out / "03/author_profiles.csv", panels)
    return outputs, panels


def main():
    """Run one bounded, resumable packet without retraining or silent rescue."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    source, bank = prepare(args.out)
    parsed = parse(args.out, bank)
    observed, declared = inputs(args.out, source, bank, parsed)
    for reader in MODELS:
        encode(args.out, reader, args.device)
    predictions, observations = evaluate(args.out, source, bank, observed, declared)
    metric_rows, alignment, comparisons = metrics(args.out, bank, observed, predictions)
    versions, panels = author_histories(args.out, bank, observations)
    write_json(
        args.out / "03/summary.json",
        {
            "materials": len(bank),
            "expected_events": sum(len(r["events"]) for r in bank),
            "actual_projections": sum(len(e) for e in observed),
            "aligned_modes": len(alignment),
            "mode_correct": sum(r["correct_mode"] for r in alignment),
            "metrics": metric_rows,
            "comparisons": comparisons,
            "version_readouts": len(versions),
            "author_profiles": len(panels),
            "source_models_refitted": 0,
            "api_calls": 0,
            "private_reads": 0,
            "state_account_not_emotion_intensity": True,
            "full_goal_complete": False,
        },
    )
    hashes = {
        str(p.relative_to(args.out)): sha(p)
        for p in sorted(args.out.rglob("*"))
        if p.is_file()
        and not p.name.startswith("._")
        and p != args.out / "04/artifact_hashes.json"
    }
    write_json(args.out / "04/artifact_hashes.json", hashes)
    print(
        f"HM-06 finished {len(predictions)} predictions; {len(hashes)} hashes",
        flush=True,
    )


if __name__ == "__main__":
    main()

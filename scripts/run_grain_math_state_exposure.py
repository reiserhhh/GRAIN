#!/usr/bin/env python3
"""HM-09B cold multi-trigger report incidence with frozen offline observers."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_local_roles import vectors
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

from grain_research.construction import math_binding as B
from grain_research.construction import math_local as L
from grain_research.construction import math_state_exposure as S
from grain_research.source_event_graph import stanza_words

SPEC = ROOT / "configs/grain_math_state_exposure_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/state_exposure_r1"


def check_source(source):
    """Verify the complete original source manifest; never refit a target."""
    hashes = json.loads((source / "04/artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(source / name) != digest:
            raise ValueError("source dependency changed: " + name)
    return sha(source / "04/artifact_hashes.json")


def prepare(out):
    """Freeze fresh public expressions, scoring rules and dependency hashes."""
    if (out / "04/summary.json").exists():
        raise ValueError("completed packet frozen; choose a new output")
    spec = json.loads(SPEC.read_text())
    source = ROOT / spec["source_packet"]
    source_hash = check_source(source)
    bank = S.material_bank(read_bank(source / "00/singles.jsonl"))
    paths = [
        Path(__file__),
        SPEC,
        ROOT / "grain_research/construction/math_state_exposure.py",
        ROOT / "grain_research/construction/math_observation_dag.py",
        ROOT / "grain_research/construction/math_purpose_accounts.py",
        ROOT / "grain_research/source_event_graph.py",
        ROOT / "scripts/run_grain_math_response.py",
        ROOT / "scripts/run_grain_math_binding.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    content = "".join(json.dumps(r) + "\n" for r in bank)
    protocol = out / "00/protocol_before_run.json"
    if protocol.exists():
        prior = json.loads(protocol.read_text())
        if (
            prior["signature"] != signature
            or prior["source_manifest"] != source_hash
            or (out / "00/documents.jsonl").read_text() != content
        ):
            raise ValueError("frozen HM-09B protocol changed")
    else:
        for p in paths:
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
                "spec": spec,
                "signature": signature,
                "source_manifest": source_hash,
                "materials": len(bank),
                "frozen_before_calls": True,
                "source_fits": 0,
                "gold_used_for_routing": False,
            },
        )
        write_json(out / "00/incidence_witness.json", S.incidence_witness())
        write_json(out / "00/joint_extension_witness.json", S.joint_extension_witness())
    print(f"HM-09B frozen {len(bank)} cold documents", flush=True)
    return source, bank


def parse(out, bank):
    """Run actual offline Stanza before consulting expected addresses or roles."""
    from fetch_grain_syntax_reference import MODELS as SYNTAX
    from fetch_grain_syntax_reference import PACKAGES, PROCESSORS
    from run_grain_source_event_graph import verify_models

    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    if stanza.__version__ != "1.11.0":
        raise ValueError("frozen parser changed")
    torch.set_num_threads(2)
    pipeline, parsed, calls, start = None, [], 0, time.monotonic()
    for i, row in enumerate(bank):
        path = out / f"01/parses/{i:04d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["text_sha256"] != sha_text(row["text"]):
                raise ValueError("raw parse text changed")
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
            data = {
                "text_sha256": sha_text(row["text"]),
                "words": words,
                **S.observe_reports(words, row["text"]),
            }
            write_json(path, data)
            calls += 1
        parsed.append(data)
        if (i + 1) % 256 == 0:
            print(
                f"HM-09B parse {i + 1}/{len(bank)} {time.monotonic() - start:.1f}s",
                flush=True,
            )
    write_json(
        out / "01/parser_accounting.json",
        {
            "new_calls": calls,
            "cached_rows": len(bank) - calls,
            "elapsed_seconds": time.monotonic() - start,
            "parser_inventory_sha256": sha(SYNTAX / "grain_inventory.json"),
        },
    )
    return parsed


def sha_text(text):
    """Hash raw expression bytes without normalization."""
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def indices(out, source, bank, parsed):
    """Address all actual views and a separate declared-reference control."""
    old = {
        r["text"]: i for i, r in enumerate(read_bank(source / "00/material_bank.jsonl"))
    }
    fresh, lookup, indexed = [], {}, []

    def ref(text):
        if text in old:
            return ["source", old[text]]
        if text not in lookup:
            lookup[text] = len(fresh)
            fresh.append({"text": text, "kind": "state_report_view"})
        return ["new", lookup[text]]

    def decorate(reports, text):
        return [
            {
                **r,
                "whole_ref": ref(r.get("projection_text", text)),
                "triggers": [
                    {
                        **t,
                        "action_ref": ref("Action: " + t["action"] + "."),
                        "object_ref": ref("Object: " + t["object"] + ".")
                        if t["object"] is not None
                        else None,
                    }
                    for t in r["triggers"]
                ],
            }
            for r in reports
        ]

    for row, data in zip(bank, parsed, strict=True):
        declared = [
            {
                **r,
                "report_id": f"declared_{j}",
                "status": "ok",
                "report_mass": 1,
                "triggers": [
                    {**t, "node_id": f"declared_{j}_{k}", "status": "ok"}
                    for k, t in enumerate(r["triggers"])
                ],
            }
            for j, r in enumerate(row["expected_reports"])
        ]
        indexed.append(
            {
                "observed": decorate(data["reports"], row["text"]),
                "declared": decorate(declared, row["text"]),
            }
        )
    content = "".join(json.dumps(r) + "\n" for r in fresh)
    path = out / "00/material_bank.jsonl"
    if path.exists() and path.read_text() != content:
        raise ValueError("frozen encoding inputs changed")
    path.write_text(content)
    write_json(out / "01/observation_indices.json", indexed)
    return indexed


def coverage_audit(bank, indexed):
    """Compare observed and expected addresses only after extraction is frozen."""
    rows = []
    for row, data in zip(bank, indexed, strict=True):
        observed = {tuple(r["root_span"]): r for r in data["observed"]}
        expected = {tuple(r["root_span"]): r for r in row["expected_reports"]}
        failures = []
        if set(observed) != set(expected):
            failures.append("report_root_addresses")
        for span in set(observed) & set(expected):
            r, gold = observed[span], expected[span]
            if (r["experiencer"], r["state_polarity"]) != (
                gold["experiencer"],
                gold["state_polarity"],
            ):
                failures.append("experiencer_or_state_negation")
            ot = {tuple(t["action_span"]): t for t in r["triggers"]}
            gt = {tuple(t["action_span"]): t for t in gold["triggers"]}
            if set(ot) != set(gt) or len(ot) != len(r["triggers"]):
                failures.append("condition_addresses")
            for at in set(ot) & set(gt):
                if any(ot[at][k] != gt[at][k] for k in ("actor", "polarity", "object")):
                    failures.append("condition_actor_sign_or_object")
            if r["status"] != "ok":
                failures.extend(r["flags"])
        rows.append(
            {
                "material_id": row["material_id"],
                "kind": row["kind"],
                "complete": not failures,
                "observed_reports": len(observed),
                "expected_reports": len(expected),
                "observed_conditions": sum(
                    len(r["triggers"]) for r in data["observed"]
                ),
                "expected_conditions": sum(
                    len(r["triggers"]) for r in row["expected_reports"]
                ),
                "failures": "|".join(sorted(set(failures))),
            }
        )
    return rows


def expected_profile(reports):
    """Evaluate declared coverage truth, never used by extraction or scoring."""
    own = [r for r in reports if r["experiencer"] == "self"]
    cap = np.array(
        [
            [
                float(
                    any(t["domain"] == target and t["relevant"] for t in r["triggers"])
                )
                for target in S.TARGETS
            ]
            for r in own
        ],
        float,
    ).reshape(-1, 2)
    return cap.mean(0).tolist() if len(cap) else None, float(
        np.min(cap, axis=1).mean()
    ) if len(cap) else None


def evaluate(out, source, bank, indexed, audit):
    """Compute fixed condition functions, report coverage and incidence contrasts."""
    conditions, readouts, predictions = [], [], []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        models = [load_model(source / f"02/{reader}_{t}.npz") for t in S.TARGETS]
        entries, report_keys, whole_refs = [], [], []
        for i, data in enumerate(indexed):
            for arm in ("observed", "declared"):
                for ri, report in enumerate(data[arm]):
                    start = len(entries)
                    entries.extend(
                        (i, arm, ri, ti, t)
                        for ti, t in enumerate(report["triggers"])
                        if t["object_ref"] is not None
                    )
                    report_keys.append((i, arm, ri, start, len(entries), report))
                    whole_refs.append(report["whole_ref"])
        a = vectors([e[-1]["action_ref"] for e in entries], old, new)
        o = vectors([e[-1]["object_ref"] for e in entries], old, new)
        score = np.column_stack(
            [B.score_kernel(m["auto_tensor"], a, o) for m in models]
        )
        whole = np.column_stack(
            [
                L.affine_score(m["whole_factors"], vectors(whole_refs, old, new))[:, 0]
                for m in models
            ]
        )
        for j, (i, arm, ri, ti, t) in enumerate(entries):
            conditions.append(
                {
                    "reader": reader,
                    "material_id": bank[i]["material_id"],
                    "arm": arm,
                    "report_index": ri,
                    "trigger_index": ti,
                    "node_id": t["node_id"],
                    "actor": t["actor"],
                    "polarity": t["polarity"],
                    "coordination_k": float(score[j, 0]),
                    "verification_k": float(score[j, 1]),
                }
            )
        by_document = defaultdict(list)
        for wi, (i, arm, ri, start, end, report) in enumerate(report_keys):
            if end - start != len(report["triggers"]):
                continue
            local = score[start:end]
            for control in ("relational", "self_trigger_only", "whole_sentence"):
                x = (
                    np.tile(whole[wi], (len(local), 1))
                    if control == "whole_sentence"
                    else local
                )
                result = S.report_capacity(
                    x, report, actor_filter=control == "self_trigger_only"
                )
                flat = {
                    "reader": reader,
                    "material_id": bank[i]["material_id"],
                    "kind": bank[i]["kind"],
                    "arm": arm,
                    "control": control,
                    "report_index": ri,
                    "report_id": report["report_id"],
                    "experiencer": report["experiencer"],
                    "state_polarity": report["state_polarity"],
                    "conditions": len(local),
                    "other_actor_conditions": sum(
                        t["actor"] == "other" for t in report["triggers"]
                    ),
                    "eligible": result["eligible"],
                    "report_mass": result["report_mass"],
                }
                for key in (
                    "coverage",
                    "hard_coverage",
                    "signed_expression",
                    "flat_sum",
                    "flat_mean",
                    "noisy_or_control",
                ):
                    for qi, target in enumerate(S.TARGETS):
                        flat[target + "_" + key] = (
                            result[key][qi] if result[key] is not None else None
                        )
                for key in ("coexposure", "hard_coexposure", "same_trigger_coexposure"):
                    flat[key] = result[key]
                readouts.append(flat)
                by_document[arm, control, i].append(flat)
        for (arm, control, i), rows in by_document.items():
            rr = [r for r in rows if r["eligible"]]
            expected, joint = expected_profile(bank[i]["expected_reports"])
            pred = {
                "reader": reader,
                "material_id": bank[i]["material_id"],
                "kind": bank[i]["kind"],
                "arm": arm,
                "control": control,
                "complete": audit[i]["complete"],
                "report_mass": len(rr),
                "trigger_mass_control": sum(r["conditions"] for r in rr),
                "expected_report_mass": sum(
                    r["experiencer"] == "self" for r in bank[i]["expected_reports"]
                ),
                "coexposure": float(np.mean([r["coexposure"] for r in rr]))
                if rr
                else None,
                "hard_coexposure": float(np.mean([r["hard_coexposure"] for r in rr]))
                if rr
                else None,
                "expected_coexposure": joint,
            }
            for qi, target in enumerate(S.TARGETS):
                for field in (
                    "coverage",
                    "hard_coverage",
                    "flat_sum",
                    "flat_mean",
                    "noisy_or_control",
                ):
                    pred[target + "_" + field] = (
                        float(np.mean([r[target + "_" + field] for r in rr]))
                        if rr
                        else None
                    )
                pred[target + "_expected"] = (
                    expected[qi] if expected is not None else None
                )
            predictions.append(pred)
    write_csv(out / "03/condition_scores.csv", conditions)
    write_csv(out / "03/report_readouts.csv", readouts)
    write_csv(out / "03/document_profiles.csv", predictions)
    contrasts = paired_contrasts(bank, predictions)
    write_csv(out / "03/paired_contrasts.csv", contrasts)
    summaries = []
    for reader, arm, control in sorted(
        {(r["reader"], r["arm"], r["control"]) for r in predictions}
    ):
        rr = [
            r
            for r in predictions
            if (r["reader"], r["arm"], r["control"]) == (reader, arm, control)
            and r["complete"]
            and r["report_mass"]
        ]
        summaries.append(
            {
                "reader": reader,
                "arm": arm,
                "control": control,
                "complete_own_documents": len(rr),
                "report_mass": sum(r["report_mass"] for r in rr),
                "trigger_mass_control": sum(r["trigger_mass_control"] for r in rr),
                "target_error_cells": sum(
                    abs(r[t + "_hard_coverage"] - r[t + "_expected"]) > 1e-12
                    for r in rr
                    for t in S.TARGETS
                ),
                "joint_error_documents": sum(
                    abs(r["hard_coexposure"] - r["expected_coexposure"]) > 1e-12
                    for r in rr
                ),
            }
        )
    write_csv(out / "03/score_summary.csv", summaries)
    return conditions, readouts, predictions, contrasts, summaries


def paired_contrasts(bank, predictions):
    """Compare report duplication and same-word report-incidence swaps."""
    lookup = {
        (r["reader"], r["arm"], r["control"], r["material_id"]): r for r in predictions
    }
    groups = defaultdict(dict)
    for r in bank:
        if r["kind"] in {"single", "duplicate"}:
            groups["duplication", r["pair_key"]][r["kind"]] = r
        if r["kind"] == "bundle":
            groups["incidence", r["bundle_key"]][r["bundled"]] = r
    rows = []
    for (kind, key), pair in groups.items():
        left, right = (
            (pair.get("single"), pair.get("duplicate"))
            if kind == "duplication"
            else (pair.get(False), pair.get(True))
        )
        if left is None or right is None:
            raise ValueError("incomplete frozen contrast")
        bag_match = Counter(re.findall(r"\w+", left["text"].lower())) == Counter(
            re.findall(r"\w+", right["text"].lower())
        )
        for reader in MODELS:
            for arm in ("observed", "declared"):
                a, b = [
                    lookup.get((reader, arm, "relational", r["material_id"]))
                    for r in (left, right)
                ]
                if a is None or b is None:
                    continue
                eligible = (
                    a["complete"]
                    and b["complete"]
                    and bool(a["report_mass"])
                    and bool(b["report_mass"])
                )
                row = {
                    "reader": reader,
                    "arm": arm,
                    "kind": kind,
                    "pair_key": key,
                    "left": left["material_id"],
                    "right": right["material_id"],
                    "complete_own_pair": eligible,
                    "same_word_bag": bag_match,
                }
                for field in (
                    "coexposure",
                    "hard_coexposure",
                    *(
                        t + "_" + f
                        for t in S.TARGETS
                        for f in (
                            "coverage",
                            "flat_sum",
                            "flat_mean",
                            "noisy_or_control",
                        )
                    ),
                ):
                    row[field + "_delta"] = b[field] - a[field] if eligible else None
                rows.append(row)
    return rows


def main():
    """Run a finite packet, preserving all unresolved inputs and frozen sources."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    source, bank = prepare(args.output)
    parsed = parse(args.output, bank)
    indexed = indices(args.output, source, bank, parsed)
    audit = coverage_audit(bank, indexed)
    write_csv(args.output / "01/coverage_audit.csv", audit)
    for reader in MODELS:
        encode(args.output, reader, args.device)
    conditions, readouts, predictions, contrasts, summaries = evaluate(
        args.output, source, bank, indexed, audit
    )
    check_source(source)
    summary = {
        "packet": "HM-09B",
        "materials": len(bank),
        "complete": sum(r["complete"] for r in audit),
        "coverage_by_kind": {
            k: {
                "total": sum(r["kind"] == k for r in audit),
                "complete": sum(r["kind"] == k and r["complete"] for r in audit),
            }
            for k in sorted({r["kind"] for r in audit})
        },
        "failures": [r for r in audit if not r["complete"]],
        "condition_score_rows": len(conditions),
        "report_readout_rows": len(readouts),
        "document_profile_rows": len(predictions),
        "score_summary": summaries,
        "source_fits": 0,
        "api_calls": 0,
        "private_reads": 0,
        "incidence_witness": S.incidence_witness(),
        "joint_extension_witness": S.joint_extension_witness(),
        "encoder_inputs_each_reader": len(
            read_bank(args.output / "00/material_bank.jsonl")
        ),
        "pair_summary": [],
    }
    for reader, arm, kind in sorted(
        {(r["reader"], r["arm"], r["kind"]) for r in contrasts}
    ):
        rr = [
            r
            for r in contrasts
            if (r["reader"], r["arm"], r["kind"]) == (reader, arm, kind)
            and r["complete_own_pair"]
        ]
        summary["pair_summary"].append(
            {
                "reader": reader,
                "arm": arm,
                "kind": kind,
                "complete_own_pairs": len(rr),
                "same_word_pairs": sum(r["same_word_bag"] for r in rr),
                "coexposure_delta_mean": float(
                    np.mean([r["coexposure_delta"] for r in rr])
                )
                if rr
                else None,
                "hard_coexposure_positive": sum(
                    r["hard_coexposure_delta"] > 0 for r in rr
                ),
                "coverage_delta_max": max(
                    (abs(r[t + "_coverage_delta"]) for r in rr for t in S.TARGETS),
                    default=0,
                ),
                "flat_sum_delta_max": max(
                    (abs(r[t + "_flat_sum_delta"]) for r in rr for t in S.TARGETS),
                    default=0,
                ),
            }
        )
    write_json(args.output / "04/summary.json", summary)
    write_json(
        args.output / "04/artifact_hashes.json",
        {
            str(f.relative_to(args.output)): sha(f)
            for f in sorted(args.output.rglob("*"))
            if f.is_file() and f.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"failures", "score_summary"}
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

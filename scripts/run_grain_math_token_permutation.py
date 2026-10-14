#!/usr/bin/env python3
"""HM13: bounded token-row permutation at fixed original block16 receivers."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_token_permutation as P
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import forward, hidden_tensor, load_observer
from scripts.run_grain_math_token_mean import checked_unit as mean_checked_unit
from scripts.run_grain_math_token_mean import family_interval
from scripts.run_grain_math_token_mean import verify_source as mean_verify_source
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_token_permutation_v1.json"
PLAN = ROOT / "docs/GRAIN_TOKEN_PERMUTATION_ACTION_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/token_permutation_action_r1"
CODE = (
    "grain_research/construction/math_token_permutation.py",
    "scripts/run_grain_math_token_permutation.py",
    "tests/test_grain_math_token_permutation.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def verify_source(out):
    """Require unchanged HM13, HM12, original interventions and compiled observations."""
    manifest = frozen_files(out)
    source = ROOT / manifest["spec"]["source_packet"]
    for name, expected in read_json(out / "00/source_hashes.json").items():
        if digest(source / name) != expected:
            raise ValueError(f"frozen HM12 artifact changed: {name}")
    mean_verify_source(source)
    return manifest


def captured_states(source_manifest, capture):
    """Load previously frozen float32 body and query states without neural replay."""
    source = ROOT / source_manifest["spec"]["source_packet"]
    canonical_query = source_manifest["spec"]["queries"][0]
    bodies, questions = {}, {}
    for row in capture:
        with np.load(source / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            if row["query"] == canonical_query:
                bodies[row["doc_id"]] = {
                    name: arrays[f"L16_{name}"].copy() for name in row["spans"]
                }
            questions[row["doc_id"], row["query"]] = arrays["L16_Q"].copy()
    return bodies, questions


def prepare(out):
    """Freeze 252 complete addresses, original references and exact state certificates."""
    if out.exists():
        raise FileExistsError("preserve prepared packet; resume instead of overwrite")
    start = time.monotonic()
    spec = read_json(SPEC)
    source = ROOT / spec["source_packet"]
    primary = mean_verify_source(source)
    ledger = read_json(
        ROOT / spec["source_verification"] / "frozen_primary_artifact_hashes.json"
    )
    for name, expected in ledger.items():
        if digest(source / name) != expected:
            raise ValueError(f"frozen HM12 artifact changed: {name}")
    if spec["queries"] != primary["spec"]["queries"]:
        raise ValueError("unchanged three queries required")
    old_design = read_json(source / "00/design.json")
    design = P.enumerate_interventions(old_design, spec)
    centered_rows = {
        (r["pair_id"], r["query"]): r for r in old_design if r["arm"] == "centered_only"
    }
    old_references = {
        (r["pair_id"], r["query"]): r for r in read_json(source / "00/references.json")
    }
    references = []
    for row in design:
        if row["arm"] != "receiver_reverse":
            continue
        key = row["pair_id"], row["query"]
        centered = centered_rows[key]
        actual = mean_checked_unit(
            source / "01/units" / f"unit_{centered['unit']:04d}.json",
            centered,
            primary["spec"]["noop_readout_tolerance"],
        )
        old = old_references[key]
        if actual["baseline"] != old["baseline"]:
            raise ValueError("original baseline mismatch")
        references.append(
            {
                **old,
                "centered": actual["native"]["conditional_score"],
                "centered_unit": centered["unit"],
            }
        )
    capture = read_json(source / "00/capture_addresses.json")
    bodies, _questions = captured_states(primary, capture)
    geometry = []
    done = set()
    for row in design:
        key = row["pair_id"], row["arm"]
        if key in done:
            continue
        done.add(key)
        h = bodies[row["receiver_doc"]][row["span"]]
        d = bodies[row["donor_doc"]][row["span"]]
        audit = P.permutation_audit(h, d, row["arm"])
        if (
            max(audit["mean_replay_max"], audit["covariance_replay_max"])
            > spec["moment_replay_tolerance"]
        ):
            raise ValueError("full-dimensional moment replay exceeded fixed tolerance")
        geometry.append({"pair_id": row["pair_id"], "arm": row["arm"], **audit})
    values = {
        "design": design,
        "capture_addresses": capture,
        "references": references,
        "geometry": geometry,
        "source_hashes": ledger,
    }
    out.mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in values.items():
        write_json(out / f"00/{name}.json", value)
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "packet": spec["packet"],
            "spec": spec,
            "alphabet": primary["alphabet"],
            "model_sha256": primary["model_sha256"],
            "source_files": {
                **primary["source_files"],
                **{name: digest(ROOT / name) for name in CODE},
            },
            "inputs": {
                f"00/{name}.json": digest(out / f"00/{name}.json") for name in values
            },
            "source_hashes_checked": len(ledger),
            "original_source_hashes_checked": primary["source_hashes_checked"],
            "frozen_before_new_calls": True,
            "selection_uses_input_metadata_only": True,
            "new_raw_texts": 0,
            "new_neural_training": False,
            "api_calls": 0,
            "private_text_reads": 0,
            "query_conditioned_receiver_states_retained": True,
        },
    )
    result = {
        "status": "PREPARED",
        "patch_units": 0,
        "patch_calls": 0,
        "seconds": 0,
        "prepare_seconds": time.monotonic() - start,
    }
    write_json(out / "run_status.json", result)
    print(
        {**result, "addresses": len(design), "state_certificates": len(geometry)},
        flush=True,
    )


def checked_unit(path, row, tolerance):
    """Check exact address and input certificates, finite arithmetic and real no-ops."""
    record = read_json(path)
    if record["address"] != row:
        raise ValueError("saved intervention address changed")
    actual = record["native"]["conditional_score"]
    replay = P.endpoint_effect(actual, record["reference"])
    for key, value in replay.items():
        if abs(record[key] - value) > 1e-12:
            raise ValueError("saved effect or endpoint bound changed")
    if (
        not record["row_multiset_exact"]
        or record["live_body_error"] != 0
        or record["live_question_error"] != 0
        or record["assigned_state_sha256"] != record["target_state_sha256"]
    ):
        raise ValueError("actual hook state/certificate mismatch")
    if row["tokens"] == 1 and (
        not record["is_exact_input_noop"] or abs(record["effect"]) > tolerance
    ):
        raise ValueError("single-token actual no-op failed")
    return record


def patch(out):
    """Run one bounded finite stage, atomically retaining every native output."""
    import torch

    manifest = verify_source(out)
    spec = manifest["spec"]
    source = ROOT / spec["source_packet"]
    previous = read_json(out / "run_status.json")
    design = read_json(out / "00/design.json")
    complete = 0
    for row in design:
        path = out / "01/units" / f"unit_{row['unit']:04d}.json"
        if path.exists():
            checked_unit(path, row, spec["noop_readout_tolerance"])
            complete += 1
    if complete == len(design):
        print({"status": "PATCH_ALREADY_COMPLETE", "units": complete}, flush=True)
        return
    capture = read_json(out / "00/capture_addresses.json")
    inputs = {(r["doc_id"], r["query"]): r for r in capture}
    bodies, questions = captured_states(read_json(source / "00/manifest.json"), capture)
    references = {
        (r["pair_id"], r["query"]): r for r in read_json(out / "00/references.json")
    }
    geometry = {
        (r["pair_id"], r["arm"]): r for r in read_json(out / "00/geometry.json")
    }
    observer = load_observer()
    write_json(out / "01/observer.json", observer.manifest())
    start = time.monotonic()

    def status(label):
        value = {
            **previous,
            "status": label,
            "patch_units": complete,
            "patch_calls": previous["patch_calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", value)
        return value

    status("PATCHING")
    try:
        for row in design:
            path = out / "01/units" / f"unit_{row['unit']:04d}.json"
            if path.exists():
                continue
            if (
                observer.calls >= spec["max_stage_calls"]
                or time.monotonic() - start >= spec["max_stage_seconds"]
            ):
                print(status("RESOURCE_BOUNDARY"), flush=True)
                return
            receiver = inputs[row["receiver_doc"], row["query"]]
            donor = bodies[row["donor_doc"]][row["span"]]
            audit = {}

            def hook(
                _module,
                _args,
                output,
                row=row,
                receiver=receiver,
                donor=donor,
                audit=audit,
            ):
                h = hidden_tensor(output)
                body_error = 0.0
                for name, span in receiver["spans"].items():
                    live = h[0, span["positions"]].detach().float().cpu().numpy()
                    body_error = max(
                        body_error,
                        float(np.max(np.abs(live - bodies[row["receiver_doc"]][name]))),
                    )
                question = h[0, -1].detach().float().cpu().numpy()
                question_error = float(
                    np.max(
                        np.abs(question - questions[row["receiver_doc"], row["query"]])
                    )
                )
                if max(body_error, question_error) > spec["live_state_tolerance"]:
                    raise ValueError("original receiver block16 state changed")
                current = h[0, row["positions"]].detach().float().cpu().numpy()
                reference, target = P.reference_and_target(current, donor, row["arm"])
                certificate = geometry[row["pair_id"], row["arm"]]
                if (
                    P.state_digest(reference) != certificate["reference_state_sha256"]
                    or P.state_digest(target) != certificate["target_state_sha256"]
                    or P.row_bytes(reference) != P.row_bytes(target)
                ):
                    raise ValueError("prepared full-state permutation changed")
                altered = h.clone()
                altered[0, row["positions"]] = torch.as_tensor(
                    target, dtype=h.dtype, device=h.device
                )
                assigned = altered[0, row["positions"]].detach().float().cpu().numpy()
                audit.update(
                    {
                        **certificate,
                        "assigned_state_sha256": P.state_digest(assigned),
                        "live_body_error": body_error,
                        "live_question_error": question_error,
                    }
                )
                return (altered, *output[1:]) if isinstance(output, tuple) else altered

            handle = observer.model.model.layers[spec["layer"]].register_forward_hook(
                hook
            )
            try:
                native = forward(observer, receiver["input_ids"], manifest["alphabet"])
            finally:
                handle.remove()
            if not audit:
                raise ValueError("declared hook did not run")
            ref = references[row["pair_id"], row["query"]]
            reference = (
                ref["baseline"] if row["arm"] == "receiver_reverse" else ref["centered"]
            )
            value = {
                "address": row,
                "native": native,
                "baseline": ref["baseline"],
                "centered_reference": ref["centered"],
                "reference": reference,
                **P.endpoint_effect(native["conditional_score"], reference),
                **audit,
            }
            write_json(path, value)
            checked_unit(path, row, spec["noop_readout_tolerance"])
            complete += 1
            if observer.calls % 50 == 0:
                print(status("PATCHING"), flush=True)
        print(status("COMPLETE_PATCH"), flush=True)
    except BaseException as error:
        result = status("FAILED")
        write_json(out / "failure.json", {**result, "error": repr(error)})
        raise


def cell_summaries(rows, keys, metrics, spec):
    """Report signed/absolute family-balanced effects without dropping any cells."""
    cells = defaultdict(list)
    for row in rows:
        cells[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, members in sorted(cells.items()):
        for metric in metrics:
            for absolute in (False, True):
                by_family = defaultdict(list)
                for row in members:
                    value = row[metric]
                    by_family[row["family"]].append(abs(value) if absolute else value)
                mean, lo, hi = family_interval(
                    [np.mean(values) for values in by_family.values()], spec
                )
                summaries.append(
                    {
                        **dict(zip(keys, key, strict=True)),
                        "metric": metric,
                        "statistic": "absolute" if absolute else "signed",
                        "pairs": len(members),
                        "families": len(by_family),
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "max_absolute": max(abs(r[metric]) for r in members),
                        "positive_pairs": sum(r[metric] > 0 for r in members),
                        "negative_pairs": sum(r[metric] < 0 for r in members),
                    }
                )
    return summaries


def evaluate(out):
    """Keep separate reference estimands, exact controls and placement interaction."""
    manifest = verify_source(out)
    spec = manifest["spec"]
    design = read_json(out / "00/design.json")
    records = [
        checked_unit(
            out / "01/units" / f"unit_{r['unit']:04d}.json",
            r,
            spec["noop_readout_tolerance"],
        )
        for r in design
    ]
    status = read_json(out / "run_status.json")
    if status["patch_calls"] != len(design) or status["status"] != "COMPLETE_PATCH":
        raise ValueError("complete actual finite calls required")
    flat = []
    grouped = defaultdict(dict)
    for record in records:
        row = record["address"]
        grouped[row["pair_id"], row["query"]][row["arm"]] = record
        flat.append(
            {
                **row,
                **{
                    k: v
                    for k, v in record.items()
                    if k not in ("address", "native", "arm", "pair_id")
                },
                "native": record["native"]["conditional_score"],
                "candidate_mass": record["native"]["candidate_mass"],
            }
        )
    contrasts = []
    for arms in grouped.values():
        receiver, centered = (arms[a] for a in spec["arms"])
        row = receiver["address"]
        contrasts.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "pair_id",
                        "family",
                        "role",
                        "axis",
                        "order",
                        "polarity",
                        "query",
                        "tokens",
                        "opportunity",
                    )
                },
                "receiver_placement_effect": receiver["effect"],
                "centered_placement_effect": centered["effect"],
                "placement_interaction": centered["effect"] - receiver["effect"],
                "hm12_centered_action": centered["centered_reference"]
                - centered["baseline"],
            }
        )
    cells = cell_summaries(
        flat, ("role", "axis", "query", "arm", "opportunity"), ("effect",), spec
    )
    details = cell_summaries(
        flat, ("role", "axis", "query", "arm", "order", "tokens"), ("effect",), spec
    )
    interactions = cell_summaries(
        contrasts,
        ("role", "axis", "query", "opportunity"),
        ("placement_interaction", "hm12_centered_action"),
        spec,
    )
    noops = [r for r in records if r["address"]["tokens"] == 1]
    summary = {
        "status": "FINITE_PACKET_COMPLETE_NOT_GLOBAL_SCOPE_COMPLETE",
        "new_actual_calls": status["patch_calls"],
        "seconds": status["seconds"],
        "paired_placement_comparisons": len(contrasts),
        "multi_token_pairs": len({r["pair_id"] for r in design if r["tokens"] > 1}),
        "single_token_controls": len(
            {r["pair_id"] for r in design if r["tokens"] == 1}
        ),
        "actual_noop_calls": len(noops),
        "noop_readout_max_error": max(abs(r["effect"]) for r in noops),
        "mean_replay_max": max(r["mean_replay_max"] for r in records),
        "covariance_replay_max": max(r["covariance_replay_max"] for r in records),
        "live_body_error_max": max(r["live_body_error"] for r in records),
        "live_question_error_max": max(r["live_question_error"] for r in records),
        "exact_multiset_checks": sum(r["row_multiset_exact"] for r in records),
        "max_set_function_error_lower_bound": max(
            r["set_function_max_error_lower_bound"] for r in records
        ),
        "human_personality_accuracy": False,
        "interval_scope": "empirical lexical families conditional on fixed source and reader; absolute CI is not a significance test",
        "all_cells": cells,
        "interaction_cells": interactions,
    }
    for name, rows in (
        ("interventions", flat),
        ("placement_contrasts", contrasts),
        ("cell_summary", cells),
        ("order_token_summary", details),
        ("interaction_summary", interactions),
    ):
        write_csv(out / f"02/{name}.csv", rows)
    write_json(out / "02/summary.json", summary)
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != out / "02/artifact_hashes.json"
        },
    )
    print(
        {
            k: v
            for k, v in summary.items()
            if k not in ("all_cells", "interaction_cells")
        },
        flush=True,
    )


def main():
    """Run one finite stage under a mutually exclusive packet lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "patch", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with collector_guard(args.out):
        {"prepare": prepare, "patch": patch, "evaluate": evaluate}[args.stage](args.out)


if __name__ == "__main__":
    main()

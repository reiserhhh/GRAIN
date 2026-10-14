#!/usr/bin/env python3
"""HM12: bounded actual mean-only/centered-only interventions on frozen token states."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_token_mean as M
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_prefix_compilation import check as compilation_check
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import forward, hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_token_mean_v1.json"
PLAN = ROOT / "docs/GRAIN_TOKEN_MEAN_ACTION_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/token_mean_action_r1"
CODE = (
    "grain_research/construction/math_token_mean.py",
    "scripts/run_grain_math_token_mean.py",
    "tests/test_grain_math_token_mean.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def verify_source(out):
    """Require unchanged HM12 inputs/code, all original artifacts and compiled functions."""
    manifest = frozen_files(out)
    source = ROOT / manifest["spec"]["source_packet"]
    ledger = read_json(out / "00/source_hashes.json")
    for name, expected in ledger.items():
        if digest(source / name) != expected:
            raise ValueError(f"original artifact changed: {name}")
    compilation_check(ROOT / manifest["spec"]["compilation_packet"], models=True)
    return manifest


def prepare(out):
    """Validate reused arrays/readouts, enumerate 768 addresses, and freeze before calls."""
    if out.exists():
        raise FileExistsError("preserve prepared packet; resume instead of overwrite")
    start = time.monotonic()
    spec = read_json(SPEC)
    source = ROOT / spec["source_packet"]
    primary = frozen_files(source)
    compilation_check(ROOT / spec["compilation_packet"], models=True)
    ledger = read_json(
        ROOT / spec["source_verification"] / "frozen_primary_artifact_hashes.json"
    )
    for name, expected in ledger.items():
        if digest(source / name) != expected:
            raise ValueError(f"original artifact changed: {name}")
    if spec["queries"] != primary["spec"]["queries"]:
        raise ValueError("unchanged three queries required")
    capture = read_json(source / "00/design.json")
    inputs = {(r["doc_id"], r["query"]): r for r in capture}
    pairs = read_json(source / "00/pairs.json")
    rows = M.enumerate_interventions(pairs, inputs, spec)
    if len(pairs) != spec["owner_pairs"] or len(rows) != spec["patch_forwards"]:
        raise ValueError("complete fixed design required")
    for role, count in spec["families_by_role"].items():
        if len({p["family"] for p in pairs if p["role"] == role}) != count:
            raise ValueError("original source/validation/application split changed")
    oldpatches = {
        (r["pair_id"], r["query"]): r
        for r in read_json(source / "00/patch_design.json")
        if r["layer"] == spec["layer"] and r["arm"] == "full_property"
    }
    references = []
    for pair in pairs:
        for query in spec["queries"]:
            address = inputs[pair["receiver_doc"], query]
            base = read_json(source / "01/units" / f"unit_{address['unit']:04d}.json")
            oldrow = oldpatches[pair["pair_id"], query]
            full = read_json(source / "03/units" / f"unit_{oldrow['unit']:05d}.json")
            if (
                base["address"] != address
                or base["body_query_error"] != 0
                or full["address"] != oldrow
                or full["live_body_error"] != 0
                or full["receiver_native"] != base["native"]["conditional_score"]
            ):
                raise ValueError("frozen original intervention replay failed")
            references.append(
                {
                    **pair,
                    "query": query,
                    "baseline": full["receiver_native"],
                    "full": full["native"]["conditional_score"],
                    "natural_donor": full["donor_native"],
                    "baseline_unit": address["unit"],
                    "full_unit": oldrow["unit"],
                }
            )
    model_files = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted(MODEL.iterdir())
        if path.is_file() and path.suffix in (".json", ".safetensors")
    }
    frozen = {
        **primary["source_files"],
        **model_files,
        **{name: digest(ROOT / name) for name in CODE},
    }
    values = {
        "design": rows,
        "capture_addresses": capture,
        "references": references,
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
            "source_files": frozen,
            "inputs": {
                f"00/{name}.json": digest(out / f"00/{name}.json") for name in values
            },
            "source_hashes_checked": len(ledger),
            "frozen_before_new_calls": True,
            "new_raw_texts": 0,
            "new_neural_training": False,
            "api_calls": 0,
            "private_text_reads": 0,
            "local_total_perturbation": True,
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
    print({**result, "addresses": len(rows), "checked_hashes": len(ledger)}, flush=True)


def checked_unit(path, row, tolerance):
    """Validate address, finite arithmetic and single-token actual no-op on resume."""
    record = read_json(path)
    if record["address"] != row:
        raise ValueError("saved intervention address changed")
    actual = record["native"]["conditional_score"] - record["baseline"]
    if not np.isfinite(actual) or abs(actual - record["action"]) > 1e-12:
        raise ValueError("saved intervention arithmetic changed")
    if record["live_body_error"] != 0 or record["live_question_error"] != 0:
        raise ValueError("live receiver does not match the frozen original state")
    if (
        row["tokens"] == 1
        and row["arm"] == "centered_only"
        and (not record["is_exact_input_noop"] or abs(actual) > tolerance)
    ):
        raise ValueError("single-token centered no-op failed")
    return record


def patch(out):
    """Perform one finite stage of actual decoder interventions; checkpoint each unit."""
    import torch

    manifest = verify_source(out)
    spec, source = manifest["spec"], ROOT / manifest["spec"]["source_packet"]
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
    references = {
        (r["pair_id"], r["query"]): r for r in read_json(out / "00/references.json")
    }
    canonical = {r["doc_id"]: r for r in capture if r["query"] == spec["queries"][0]}
    bodies, questions = {}, {}
    for doc_id, row in canonical.items():
        with np.load(source / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc_id] = {
                name: arrays[f"L16_{name}"].astype(float) for name in row["spans"]
            }
    for row in capture:
        with np.load(source / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            questions[row["doc_id"], row["query"]] = arrays["L16_Q"].astype(float)
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
                receiver=receiver,
                row=row,
                donor=donor,
                audit=audit,
            ):
                h = hidden_tensor(output)
                body_error = 0.0
                for name, span in receiver["spans"].items():
                    current = h[0, span["positions"]].detach().float().cpu().numpy()
                    body_error = max(
                        body_error,
                        float(
                            np.max(np.abs(current - bodies[row["receiver_doc"]][name]))
                        ),
                    )
                question = h[0, -1].detach().float().cpu().numpy()
                question_error = float(
                    np.max(
                        np.abs(question - questions[row["receiver_doc"], row["query"]])
                    )
                )
                current = h[0, row["positions"]].detach().float().cpu().numpy()
                target = M.patch_target(current, donor, row["arm"])
                audit.update(
                    {
                        **M.geometry_audit(current, donor, row["arm"]),
                        "live_body_error": body_error,
                        "live_question_error": question_error,
                    }
                )
                if max(body_error, question_error) > spec["live_state_tolerance"]:
                    raise ValueError("original receiver block16 state changed")
                altered = h.clone()
                altered[0, row["positions"]] = torch.as_tensor(
                    target, dtype=h.dtype, device=h.device
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
            value = {
                "address": row,
                "native": native,
                "baseline": ref["baseline"],
                "full_reference": ref["full"],
                "action": native["conditional_score"] - ref["baseline"],
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


def family_interval(values, spec):
    """Bootstrap lexical-family means, conditional on frozen sources and the reader."""
    x = np.asarray(values, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite family readouts required")
    rng = np.random.default_rng(spec["bootstrap_seed"])
    means = x[rng.integers(0, len(x), (spec["bootstrap_replicates"], len(x)))].mean(
        axis=1
    )
    return float(x.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def evaluate(out):
    """Keep every role/order/query and token opportunity; report mixed rather than delete it."""
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
        raise ValueError("actual full bounded-call completion required")
    grouped = defaultdict(dict)
    flat = []
    for record in records:
        r = record["address"]
        grouped[r["pair_id"], r["query"]][r["arm"]] = record
        flat.append(
            {
                **r,
                **{k: v for k, v in record.items() if k not in ("address", "native")},
                "native": record["native"]["conditional_score"],
                "candidate_mass": record["native"]["candidate_mass"],
            }
        )
    decompositions, bindings, summaries, opportunity = [], [], [], []
    contrast_groups = defaultdict(list)
    for (_pid, query), arms in grouped.items():
        m, c = arms["mean_only"], arms["centered_only"]
        r = m["address"]
        split = M.finite_response_split(
            m["baseline"],
            m["full_reference"],
            m["native"]["conditional_score"],
            c["native"]["conditional_score"],
        )
        row = {
            **{
                k: r[k]
                for k in (
                    "pair_id",
                    "family",
                    "role",
                    "axis",
                    "order",
                    "polarity",
                    "opportunity",
                    "tokens",
                )
            },
            "query": query,
            **split,
            "centered_mean_shift_norm": c["rounded_mean_shift_norm"],
        }
        decompositions.append(row)
        contrast_groups[r["family"], r["axis"], r["order"], query].append(row)
    metrics = (
        "full_action",
        "mean_action",
        "centered_action",
        "mixed_action",
        "full_minus_mean",
    )
    for rows in contrast_groups.values():
        if len(rows) != 2 or {r["polarity"] for r in rows} != {0, 1}:
            raise ValueError("complete two-polarity binding rectangle required")
        high, low = sorted(rows, key=lambda r: r["polarity"], reverse=True)
        bindings.append(
            {
                **{k: high[k] for k in ("family", "role", "axis", "order", "query")},
                "high_tokens": high["tokens"],
                "low_tokens": low["tokens"],
                "opportunity": "single"
                if high["tokens"] == low["tokens"] == 1
                else "both_multi"
                if min(high["tokens"], low["tokens"]) > 1
                else "mixed",
                **{k: high[k] - low[k] for k in metrics},
            }
        )
    cells = defaultdict(list)
    for row in bindings:
        cells[row["role"], row["axis"], row["order"], row["query"]].append(row)
    for key, rows in cells.items():
        for metric in metrics:
            mean, lo, hi = family_interval([r[metric] for r in rows], spec)
            summaries.append(
                {
                    "role": key[0],
                    "axis": key[1],
                    "order": key[2],
                    "query": key[3],
                    "metric": metric,
                    "families": len(rows),
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    cells = defaultdict(list)
    for row in decompositions:
        cells[row["role"], row["axis"], row["query"], row["opportunity"]].append(row)
    for key, rows in cells.items():
        for metric in metrics:
            by_family = defaultdict(list)
            for row in rows:
                by_family[row["family"]].append(abs(row[metric]))
            mean, lo, hi = family_interval(
                [np.mean(v) for v in by_family.values()], spec
            )
            opportunity.append(
                {
                    "role": key[0],
                    "axis": key[1],
                    "query": key[2],
                    "opportunity": key[3],
                    "metric": metric,
                    "pairs": len(rows),
                    "families": len(by_family),
                    "mean_absolute": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "max_absolute": max(abs(r[metric]) for r in rows),
                }
            )
    noop = [
        r
        for r in records
        if r["address"]["tokens"] == 1 and r["address"]["arm"] == "centered_only"
    ]
    centered = [r for r in records if r["address"]["arm"] == "centered_only"]
    summary = {
        "status": "FINITE_PACKET_COMPLETE_NOT_GLOBAL_SCOPE_COMPLETE",
        "new_actual_calls": status["patch_calls"],
        "seconds": status["seconds"],
        "paired_query_decompositions": len(decompositions),
        "binding_rectangles": len(bindings),
        "single_token_noops": len(noop),
        "noop_max_readout_error": max((abs(r["action"]) for r in noop), default=None),
        "centered_mean_shift_norm_max": max(
            r["rounded_mean_shift_norm"] for r in centered
        ),
        "live_body_error_max": max(r["live_body_error"] for r in records),
        "live_question_error_max": max(r["live_question_error"] for r in records),
        "ideal_split_error_max": max(r["ideal_split_error"] for r in records),
        "human_personality_accuracy": False,
        "interval_scope": "empirical lexical families conditional on fixed source and reader",
        "all_binding_cells": summaries,
        "opportunity_cells": opportunity,
    }
    for name, value in (
        ("interventions", flat),
        ("response_decomposition", decompositions),
        ("binding_by_family", bindings),
        ("binding_summary", summaries),
        ("opportunity_summary", opportunity),
    ):
        write_csv(out / f"02/{name}.csv", value)
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
            if k not in ("all_binding_cells", "opportunity_cells")
        },
        flush=True,
    )


def main():
    """Run exactly one bounded stage under a mutually exclusive packet lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "patch", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with collector_guard(args.out):
        {"prepare": prepare, "patch": patch, "evaluate": evaluate}[args.stage](args.out)


if __name__ == "__main__":
    main()

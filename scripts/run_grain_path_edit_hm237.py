#!/usr/bin/env python3
"""Execute one SOURCE/gap-addressed evidence edit, not a prediction search."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_path_edit as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_path_edit_hm237.json"
DEFAULT = ROOT / "results/grain_math_response_v1/path_edit_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PATH_EDIT_PLAN_HM237.md",
    "grain_research/construction/math_path_edit.py",
    "scripts/run_grain_path_edit_hm237.py",
    "tests/test_grain_path_edit.py",
    "grain_research/construction/math_path_functional.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
)
STAGES = (
    "compiled_seal.json",
    "cold_seal.json",
    "analyzed_seal.json",
    "verification_seal.json",
)


def clean(value):
    """Serialize unsupported values as null, not zero or a neutral prediction."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def prepare(out):
    """Authenticate exact existing events without reading application outcomes."""
    cfg = R.read(ROOT / CONFIG)
    sources = {
        k: ROOT / cfg[k]
        for k in (
            "source_parent",
            "path_parent",
            "event_parent",
            "meaning_parent",
            "native_parent",
        )
    }
    files = []
    for k, names in (
        ("source_parent", ("source_inputs.npz", "input_quality.json")),
        ("path_parent", ("observations.npz",)),
        ("event_parent", ("observations.npz",)),
        ("meaning_parent", ("observations.npz", "bindings.json")),
        ("native_parent", ("observations.npz", "selections.json")),
    ):
        S.parent_seal(sources[k], "prepared_seal.json", names)
        files.extend(sources[k] / p for p in ("prepared_seal.json",) + names)
    native = sources["native_parent"]
    S.parent_seal(native, "analyzed_seal.json", ("analysis.npz",))
    files.extend(native / p for p in ("analyzed_seal.json", "analysis.npz"))
    source = S.selected(
        sources["source_parent"] / "source_inputs.npz", ("source_paths", "targets")
    )
    S.parent_seal(sources["source_parent"], "compiled_seal.json", ("cold_bundle.npz",))
    baseline = S.selected(
        sources["source_parent"] / "cold_bundle.npz", ("point_field_0", "point_field_1")
    )
    files.extend(
        sources["source_parent"] / p for p in ("compiled_seal.json", "cold_bundle.npz")
    )
    source_manifest = R.read(sources["source_parent"] / "manifest.json")
    source_address = ROOT / source_manifest["config"]["parent"]
    S.parent_seal(source_address, "prepared_seal.json", ("observations.npz",))
    addresses = S.selected(
        source_address / "observations.npz", ("source_indices", "held")
    )
    files.extend(source_address / p for p in ("prepared_seal.json", "observations.npz"))
    paths = S.selected(sources["path_parent"] / "observations.npz", ("paths", "held"))
    events = S.selected(
        sources["event_parent"] / "observations.npz", ("times", "original_units")
    )
    meaning = S.selected(
        sources["meaning_parent"] / "observations.npz",
        ("cells", "source_indices", "application_indices"),
    )
    data = S.selected(native / "observations.npz", ("panel_indices", "source_indices"))
    selections = R.read(native / "selections.json")
    bindings = R.read(sources["meaning_parent"] / "bindings.json")
    src, held, panel = (
        addresses["source_indices"],
        addresses["held"],
        data["panel_indices"],
    )
    if (
        source["source_paths"].shape != (386, 2, 18)
        or source["targets"].shape != (2, 386, 5)
        or paths["paths"].shape != (624, 2, 18)
        or not np.array_equal(src, data["source_indices"])
        or not np.array_equal(src, meaning["source_indices"])
        or not np.array_equal(panel, meaning["application_indices"])
        or not np.array_equal(held, paths["held"])
        or not np.array_equal(source["source_paths"], paths["paths"][src])
        or len(src) != 386
        or len(panel) != 24
        or len(held) != 120
        or np.intersect1d(src, np.r_[held, panel]).size
    ):
        raise ValueError("Different SOURCE/observations/author identities")
    positions = np.argsort(events["original_units"], axis=1)[panel, 24:36]
    cells = np.take_along_axis(paths["paths"].reshape(624, 36)[panel], positions, 1)
    times = np.take_along_axis(events["times"][panel], positions, 1)
    if not np.array_equal(cells, meaning["cells"]) or np.any(
        np.diff(positions, axis=1) != 1
    ):
        raise ValueError("Native12 events do not map exactly to inherited paths")
    if len(selections) != 24 or len(bindings) != 24:
        raise ValueError("Full SAME-event panel required")
    records = []
    for u, original in enumerate(panel):
        sel, bind = selections[u], bindings[u]
        if (
            sel["author"] != u
            or sel["original_author_index"] != int(original)
            or bind["event_ids"] != sel["event_ids"]
        ):
            raise ValueError("Changed exact comment binding")
        ids = sel["event_ids"]
        records.append(
            {
                "author_key": f"native/{u}",
                "events": [
                    {
                        "event_id": ids[t],
                        "position": int(positions[u, t]),
                        "timestamp": float(times[u, t]),
                        "cell": int(cells[u, t]),
                    }
                    for t in range(12)
                ],
                "edits": {k: [ids[i] for i in sel[k]] for k in ("support", "counter")},
            }
        )
    for u in held:
        for w in (0, 1):
            offset = 18 * w
            ids = [
                f"event/{int(u)}/{int(events['original_units'][u, offset + t])}"
                for t in range(18)
            ]
            records.append(
                {
                    "author_key": f"held/{int(u)}/window/{w}",
                    "events": [
                        {
                            "event_id": ids[t],
                            "position": offset + t,
                            "timestamp": float(events["times"][u, offset + t]),
                            "cell": int(paths["paths"][u, w, t]),
                        }
                        for t in range(18)
                    ],
                    "edits": {
                        k: [ids[i] for i in cfg[f"held_{k}"]]
                        for k in ("support", "counter")
                    },
                }
            )
    for rec in records:
        M.validate_record(rec)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
            "cohort_reuse": "historical same-event/held cohorts and neural encoder reused",
        },
    )
    R.archive(out / "source_inputs.npz", **source, **baseline)
    with (out / "cold_inputs.jsonl").open("x") as f:
        for rec in records:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
    quality = {
        "source_authors": 386,
        "held_authors": 120,
        "held_windows": 240,
        "native_authors": 24,
        "native_events": 288,
        "panel_held_author_overlap": int(np.intersect1d(panel, held).size),
        "all_same_event_states_exact": True,
        "source_application_overlap": 0,
        "native_outcome_values_read_before_cold": False,
        "new_neural_API_private_train_application_label_calls": 0,
        "sampled_publication_not_session_state": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("source_inputs.npz", "cold_inputs.jsonl", "input_quality.json"),
    )
    print(json.dumps(quality), flush=True)


def compile_source(out):
    """Freeze the full observed-gap family before any application operation."""
    cfg = R.check(out)["config"]
    z = S.selected(
        out / "source_inputs.npz",
        ("source_paths", "targets", "point_field_0", "point_field_1"),
    )
    start = time.perf_counter()
    laws = [M.compile_source(z["source_paths"], y) for y in z["targets"]]
    baseline_errors = []
    for i, law in enumerate(laws):
        old = z[f"point_field_{i}"]
        new = law["fields"][0]
        if not np.array_equal(np.isfinite(new), np.isfinite(old)):
            raise ValueError("Original adjacent-field support changed")
        error = float(np.max(abs(new[np.isfinite(new)] - old[np.isfinite(old)])))
        if error > cfg["audit_tolerance"]:
            raise ValueError("Original SOURCE adjacent-field values changed")
        baseline_errors.append(error)
    arrays, rows = {}, []
    for i, law in enumerate(laws):
        arrays.update({f"{k}_{i}": v for k, v in law.items() if k != "tables"})
        for j, gap in enumerate(law["gaps"]):
            rows.append(
                {
                    "law": cfg["laws"][i],
                    "d1": int(gap[0]),
                    "d2": int(gap[1]),
                    "supported_cells": int(np.isfinite(law["fields"][j]).all(0).sum()),
                    "total_mass": float(law["tables"][j].sum()),
                }
            )
    seconds = time.perf_counter() - start
    if seconds > cfg["max_cpu_seconds"]:
        raise TimeoutError("Finite source cap exceeded;no retry")
    R.archive(out / "source_tables.npz", tables=np.array([l["tables"] for l in laws]))
    R.archive(out / "cold_bundle.npz", **arrays)
    R.table(out / "source_gap_accounts.csv", rows)
    R.write(
        out / "compile_receipt.json",
        {
            "CPU_wall_seconds": seconds,
            "gaps": 15,
            "original_field_replay_errors": baseline_errors,
            "new_model_calls": 0,
        },
    )
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_tables.npz",
            "cold_bundle.npz",
            "source_gap_accounts.csv",
            "compile_receipt.json",
        ),
    )
    print("SOURCE_COMPILED", seconds, flush=True)


def cold_runtime(bundle, inputs, output):
    """Read only aggregate fields and observed events, not individual SOURCE rows."""
    with np.load(bundle, allow_pickle=False) as z:
        laws = [{k: z[f"{k}_{i}"] for k in ("fields", "gaps", "prior")} for i in (0, 1)]
    with inputs.open() as f, output.open("x") as dest:
        for line in f:
            rec = json.loads(line)
            value = {
                "author_key": rec["author_key"],
                "laws": [M.edit_accounts(**l, record=rec) for l in laws],
            }
            dest.write(json.dumps(clean(value), allow_nan=False) + "\n")


def cold(out):
    """Actually run an isolated numeric-only scoring process once."""
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    start = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "cold-runtime",
            "--bundle",
            str(out / "cold_bundle.npz"),
            "--inputs",
            str(out / "cold_inputs.jsonl"),
            "--output",
            str(out / "cold_outputs.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=cfg["max_cpu_seconds"],
        check=False,
    )
    R.write(
        out / "cold_receipt.json",
        {
            "command": proc.args,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "CPU_wall_seconds": time.perf_counter() - start,
            "individual_SOURCE_or_application_labels_read": False,
            "new_model_calls": 0,
        },
    )
    proc.check_returncode()
    R.seal(out, "cold_seal.json", ("cold_outputs.jsonl", "cold_receipt.json"))
    print("COLD_COMPLETE", flush=True)


def output_rows(out):
    """Load the complete frozen cold outputs, retaining unsupported entries."""
    return [
        json.loads(line)
        for line in (out / "cold_outputs.jsonl").read_text().splitlines()
    ]


def association(x, y):
    """Descriptive SAME-event correlation, never a calibrated psychological score."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def analyze(out):
    """Keep all laws, targets, signed accounts and old native contacts."""
    cfg = R.check(out, ("compiled_seal.json", "cold_seal.json"))["config"]
    all_rows = output_rows(out)
    if len(all_rows) != 264 or len({r["author_key"] for r in all_rows}) != 264:
        raise ValueError("Full264 history scope required")
    detailed, summaries, ledgers, contacts = [], [], [], []
    native = S.selected(
        ROOT / cfg["native_parent"] / "analysis.npz", ("native_readings",)
    )["native_readings"]
    if native.shape != (24, 3, 5):
        raise ValueError("All SAME24-author/native5 targets required")
    native_action = native[:, 2] - native[:, 1]
    for law, label in enumerate(cfg["laws"]):
        for row in all_rows:
            r = row["laws"][law]
            for entry in r["ledger"]:
                ledgers.append({"author_key": row["author_key"], "law": label, **entry})
            for q, target in enumerate(cfg["targets"]):
                parts = np.array(r["part_actions"], float)[:, q]
                detailed.append(
                    {
                        "author_key": row["author_key"],
                        "law": label,
                        "target": target,
                        "blind_action": r["blind_action"][q],
                        "known_gap_action": r["known_gap_action"][q],
                        "destroyed_action": clean(parts[0]),
                        "splice_action": clean(parts[1]),
                        "normalization_action": clean(parts[2]),
                        "gap_law_action": clean(parts[3]),
                    }
                )
        for group in ("native", "held"):
            subset = [r for r in all_rows if r["author_key"].startswith(group + "/")]
            for q, target in enumerate(cfg["targets"]):
                blind = np.array(
                    [r["laws"][law]["blind_action"][q] for r in subset], float
                )
                known = np.array(
                    [r["laws"][law]["known_gap_action"][q] for r in subset], float
                )
                parts = np.array(
                    [r["laws"][law]["part_actions"] for r in subset], float
                )[:, :, q]
                valid = (
                    np.isfinite(blind) & np.isfinite(known) & np.isfinite(parts).all(1)
                )
                finite = parts[valid]
                summaries.append(
                    {
                        "law": label,
                        "group": group,
                        "target": target,
                        "histories": len(subset),
                        "both_action_supported": int(valid.sum()),
                        "sign_disagreements": int(
                            (blind[valid] * known[valid] < 0).sum()
                        ),
                        "blind_RMS": float(np.sqrt(np.mean(blind[valid] ** 2)))
                        if valid.any()
                        else None,
                        "known_gap_RMS": float(np.sqrt(np.mean(known[valid] ** 2)))
                        if valid.any()
                        else None,
                        **{
                            name + "_RMS": float(np.sqrt(np.mean(finite[:, i] ** 2)))
                            if valid.any()
                            else None
                            for i, name in enumerate(
                                ("destroyed", "splice", "normalization", "gap_law")
                            )
                        },
                    }
                )
                if group == "native":
                    for nq, nt in enumerate(cfg["targets"]):
                        contacts.append(
                            {
                                "law": label,
                                "source_target": target,
                                "native_target": nt,
                                "blind_contact": association(
                                    blind, native_action[:, nq]
                                ),
                                "known_gap_contact": association(
                                    known, native_action[:, nq]
                                ),
                            }
                        )
    R.table(out / "edit_actions.csv", detailed)
    R.table(out / "action_summary.csv", summaries)
    R.table(out / "native_contacts.csv", contacts)
    with (out / "edit_ledger.jsonl").open("x") as f:
        for rec in ledgers:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
    topology = []
    for r in all_rows:
        for name, arm in r["laws"][0]["arms"].items():
            topology.append(
                {
                    "author_key": r["author_key"],
                    "arm": name,
                    **{
                        k: arm[k]
                        for k in (
                            "retained",
                            "destroyed",
                            "created",
                            "old_triples",
                            "edited_triples",
                            "blind_triple_coverage",
                            "known_gap_triple_coverage",
                        )
                    },
                }
            )
    R.table(out / "topology.csv", topology)
    summary = {
        "status": "ALL_SCOPE_ANALYZED",
        "histories": 264,
        "all_accounts": summaries,
        "mean_created_triples": float(np.mean([r["created"] for r in topology])),
        "zero_retained_edit_histories": sum(r["retained"] == 0 for r in topology),
        "mean_known_gap_coverage": float(
            np.mean([r["known_gap_triple_coverage"] for r in topology])
        ),
        "all_native_contacts_kept": True,
        "prediction_promotion": False,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "edit_actions.csv",
            "action_summary.csv",
            "native_contacts.csv",
            "edit_ledger.jsonl",
            "topology.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Direct alternative source counts and original/edited score identities."""
    cfg = R.check(out, STAGES[:3])["config"]
    z = S.selected(out / "source_inputs.npz", ("source_paths", "targets"))
    tables = S.selected(out / "source_tables.npz", ("tables",))["tables"]
    gaps = M.gap_patterns()
    direct = np.zeros_like(tables)
    for u, windows in enumerate(z["source_paths"]):
        codes = z["targets"][:, u].astype(int) @ (1 << np.arange(5))
        for path in windows:
            for j, (a, b) in enumerate(gaps):
                for t in range(18 - a - b):
                    for l, code in enumerate(codes):
                        direct[l, j, code, path[t], path[t + a], path[t + a + b]] += (
                            1 / (386 * 2 * (18 - a - b))
                        )
    source_error = float(np.max(abs(direct - tables)))
    records = [
        json.loads(s) for s in (out / "cold_inputs.jsonl").read_text().splitlines()
    ]
    rows = output_rows(out)
    bundle = S.selected(
        out / "cold_bundle.npz",
        tuple(f"{k}_{i}" for i in (0, 1) for k in ("fields", "prior")),
    )
    error, checked, unsupported = 0.0, 0, 0
    for rec, row in zip(records, rows, strict=True):
        if rec["author_key"] != row["author_key"]:
            raise ValueError("Cold identity mismatched")
        for l in (0, 1):
            field, prior = bundle[f"fields_{l}"], bundle[f"prior_{l}"]
            result = row["laws"][l]
            events = rec["events"]
            ids = [e["event_id"] for e in events]
            full = np.mean(
                [
                    field[
                        0,
                        :,
                        events[t]["cell"],
                        events[t + 1]["cell"],
                        events[t + 2]["cell"],
                    ]
                    - prior
                    for t in range(len(events) - 2)
                ],
                axis=0,
            )
            if np.isfinite(full).all():
                error = max(
                    error, float(np.max(abs(full - np.array(result["full"], float))))
                )
            for name, arm in result["arms"].items():
                kept = [i for i, e in enumerate(ids) if e not in rec["edits"][name]]
                vals, known_vals = [], []
                for t in range(len(kept) - 2):
                    a, b, c = kept[t : t + 3]
                    cells = [events[i]["cell"] for i in (a, b, c)]
                    j = next(
                        i for i, d in enumerate(gaps) if tuple(d) == (b - a, c - b)
                    )
                    vals.append(field[0, :, *cells] - prior)
                    known_vals.append(field[j, :, *cells] - prior)
                blind, known = np.mean(vals, 0), np.mean(known_vals, 0)
                parts = np.array(arm["parts"], float)
                for actual, saved in ((blind, arm["blind"]), (known, arm["known_gap"])):
                    saved = np.array(saved, float)
                    if not np.array_equal(np.isfinite(actual), np.isfinite(saved)):
                        raise ValueError("Unsupported value was rescued or dropped")
                    if np.isfinite(actual).all():
                        error = max(error, float(np.max(abs(actual - saved))))
                if (
                    np.isfinite(full).all()
                    and np.isfinite(blind).all()
                    and np.isfinite(known).all()
                ):
                    error = max(
                        error,
                        float(np.max(abs(parts[:3].sum(0) - (blind - full)))),
                        float(np.max(abs(parts.sum(0) - (known - full)))),
                    )
                    checked += 1
                else:
                    unsupported += 1
    if max(error, source_error) > cfg["audit_tolerance"]:
        raise ValueError("Literal/source edit accounts disagree;preserve failure")
    receipt = {
        "PASS": True,
        "source_literal_error": source_error,
        "edit_account_error": error,
        "complete_law_history_edit_accounts": checked,
        "incomplete_accounts_retained": unsupported,
        "same_agent_not_blind": True,
        "new_neural_API_private_train_calls": 0,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt), flush=True)


def main():
    """Run exactly one immutable finite stage or the isolated cold CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=(
            "prepare",
            "compile",
            "cold",
            "analyze",
            "verify",
            "check",
            "cold-runtime",
        ),
    )
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.stage == "cold-runtime":
        cold_runtime(args.bundle, args.inputs, args.output)
    elif args.stage == "check":
        R.check(args.out, STAGES)
        print("ALL_SEALS_PASS", flush=True)
    else:
        action = {
            "prepare": prepare,
            "compile": compile_source,
            "cold": cold,
            "analyze": analyze,
            "verify": verify,
        }[args.stage]
        try:
            action(args.out)
        except Exception as exc:
            if args.out.exists():
                failure = args.out / (args.stage + "_failure.json")
                if not failure.exists():
                    R.write(
                        failure,
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "no_auto_retry": True,
                        },
                    )
            raise


if __name__ == "__main__":
    main()

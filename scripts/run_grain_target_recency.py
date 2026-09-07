#!/usr/bin/env python3
"""P1M: explicit query targets and action-record order, with paired state patches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.role_opportunity import check_donor_layout
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_evidence_conduit import partitions
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_vector_mechanism_controls import write_json


def reorder(evidence, order):
    """Reverse exactly two named action lines, never their contents or ownership."""
    lines = evidence.splitlines()
    if len(lines) != 3 or {v.split()[0] for v in lines[1:]} != {"Alex", "Blair"}:
        raise ValueError("Expected header plus two named action lines")
    if order not in ("original", "reversed"):
        raise ValueError("Unknown record order")
    return "\n".join(lines[:1] + (lines[1:] if order == "original" else lines[:0:-1]))


def material(parent):
    """Keep all old cases in both orders and add one room-target case per record."""
    rows, seen = [], set()
    for r in parent:
        for order in ("original", "reversed"):
            evidence = reorder(r["evidence"], order)
            suffix = r["user"][len(r["evidence"]) :]
            user = evidence + suffix
            shift = len(evidence) - len(r["evidence"])
            rows.append(
                {
                    **r,
                    "case_id": r["case_id"] + "__" + order,
                    "parent_case": r["case_id"],
                    "record_order": order,
                    "evidence": evidence,
                    "user": user,
                    "anchor_span": [i + shift for i in r["entity_span"]],
                    "anchor_text": r["entity"],
                    "target_kind": "person",
                    "last_named": evidence.splitlines()[-1].split()[0],
                }
            )
            key = (r["group_id"], r["level"], order)
            if key in seen:
                continue
            seen.add(key)
            prefix = evidence + "\n\nQuestion: For the"
            start = len(prefix)
            prefix += " room"
            rows.append(
                {
                    "case_id": f"{r['group_id']}__{r['level']}__room__venue_object__{order}",
                    "parent_case": r["case_id"],
                    "group_id": r["group_id"],
                    "level": r["level"],
                    "record_order": order,
                    "entity": "room",
                    "criterion": "venue_object",
                    "target_kind": "object",
                    "role_sign": r["level"],
                    "expected_sign": 1,
                    "evidence": evidence,
                    "anchor_span": [start, len(prefix)],
                    "anchor_text": "room",
                    "last_named": evidence.splitlines()[-1].split()[0],
                    "user": prefix
                    + ", was the room open during the event? Return A for Yes and B for No.",
                }
            )
    if (
        len(rows) != 112
        or len({r["case_id"] for r in rows}) != 112
        or len({r["user"] for r in rows}) != 112
    ):
        raise ValueError("Expected112 distinct prompt cells")
    return rows


def summarize(base, effects):
    """Expose raw venue leakage and per-person record-order contrasts without pooling people."""
    lookup = base.set_index("case_id").margin
    effects = effects.copy()
    effects["shift"] = effects.margin - effects.case_id.map(lookup)
    effects["gap"] = effects.donor_case.map(lookup) - effects.case_id.map(lookup)
    effects["restore_error"] = effects["shift"] - effects.gap
    keys = ["group_id", "entity", "criterion", "record_order"]
    pairs = []
    for key, sub in base.groupby(keys):
        x = sub.set_index("role_sign").margin
        pairs.append(
            dict(zip(keys, key))
            | {"channel": "native", "contrast": float(x[1] - x[-1])}
        )
    for key, sub in effects.groupby(keys + ["patch"]):
        pairs.append(
            dict(zip(keys + ["channel"], key))
            | {"contrast": float(np.mean(-sub.role_sign * sub["shift"]))}
        )
    pairs = pd.DataFrame(pairs)
    summary = []
    for (criterion, order, channel), sub in pairs.groupby(
        ["criterion", "record_order", "channel"]
    ):
        native = base[(base.criterion == criterion) & (base.record_order == order)]
        summary.append(
            {
                "criterion": criterion,
                "record_order": order,
                "channel": channel,
                "pairs": len(sub),
                "mean_contrast": float(sub.contrast.mean()),
                "rms_contrast": float(np.sqrt(np.mean(sub.contrast**2))),
                "positive_role_pairs": int(np.sum(sub.contrast > 0))
                if criterion in ("event", "trait")
                else None,
                "native_expected_preference": float(
                    np.mean(native.margin * native.expected_sign > 0)
                ),
            }
        )
    order_rows = []
    for key, sub in pairs.groupby(["group_id", "entity", "criterion", "channel"]):
        values = sub.set_index("record_order").contrast
        order_rows.append(
            dict(zip(["group_id", "entity", "criterion", "channel"], key))
            | {
                "original": float(values.original),
                "reversed": float(values.reversed),
                "order_delta": float(values.reversed - values.original),
                "same_sign": bool(values.original * values.reversed > 0),
            }
        )
    venue = []
    for key, sub in pairs[pairs.criterion.isin(["venue", "venue_object"])].groupby(
        ["group_id", "record_order", "channel"]
    ):
        person, room = (
            sub[sub.criterion == "venue"],
            sub[sub.criterion == "venue_object"],
        )
        if len(person) != 2 or len(room) != 1:
            raise ValueError("Room control double-counted or missing")
        rms = float(np.sqrt(np.mean(person.contrast**2)))
        value = float(abs(room.contrast.iloc[0]))
        venue.append(
            dict(zip(["group_id", "record_order", "channel"], key))
            | {
                "person_rms": rms,
                "object_abs": value,
                "object_minus_person": value - rms,
            }
        )
    return {
        "response_rows": effects,
        "role_pairs": pairs,
        "summary": pd.DataFrame(summary),
        "order_contrasts": pd.DataFrame(order_rows),
        "venue_comparison": pd.DataFrame(venue),
    }


def run(args):
    """Run one fixed-layer complete packet and preserve raw incomplete work on failure."""
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    if (
        json.loads((args.parent_dir / "run_status.json").read_text())["status"]
        != "COMPLETE"
    ):
        raise ValueError("Complete P1J parent required")
    for name, digest in json.loads(
        (args.parent_dir / "output_hashes.json").read_text()
    ).items():
        if file_hash(args.parent_dir / name) != digest:
            raise ValueError("Changed parent artifact")
    data = json.loads((args.parent_dir / "materials.json").read_text())
    rows = material(data["rows"])
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "activations").mkdir()
    sources = list(
        dict.fromkeys(
            [
                Path(__file__),
                ROOT / "docs/GRAIN_TARGET_RECENCY_P1M_PLAN.md",
                ROOT / "tests/test_grain_target_recency.py",
            ]
            + [ROOT / p for p in parent["code"]]
        )
    )
    manifest = {
        "parent_identity": parent["identity"],
        "material_hash": material_hash(rows),
        "max_calls": 400,
        "max_seconds": 600,
        "pad_to_length": 256,
        "private_data_read": False,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": data["system"], "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, p.relative_to(ROOT))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    base, effects, controls = [], [], []
    observer, start = None, time.monotonic()

    def flush(status, **extra):
        """Keep raw forward count and outputs, not a misleading partial summary."""
        pd.DataFrame(base).to_csv(out / "baselines.csv", index=False)
        pd.DataFrame(effects).to_csv(out / "effects.csv", index=False)
        write_json(out / "instrument_checks.json", controls)
        value = {
            "status": status,
            "identity": manifest["identity"],
            "calls": observer.calls if observer else 0,
            "seconds": time.monotonic() - start,
            "baselines": len(base),
            "effects": len(effects),
            **extra,
        }
        write_json(out / "run_status.json", value)
        return value

    try:
        expected = EXPECTED_METADATA | {
            "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        }
        if {k: file_hash(args.model_dir / k) for k in expected} != expected:
            raise ValueError("Model identity changed")
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(args.model_dir),
            device="mps",
            dtype="float32",
            max_tokens=256,
            pad_to_length=256,
        )
        observer.sites = {"block_16": observer.model.model.layers[16]}
        write_json(out / "model_manifest.json", observer.manifest())
        keys = ("group_id", "level", "entity", "criterion", "record_order")
        lookup = {tuple(r[k] for k in keys): r["case_id"] for r in rows}
        layouts, states = {}, {}
        for r in rows:
            ids = observer.prompt_ids(data["system"], r["user"])
            span = observer.content_indices(data["system"], r["user"], r["evidence"])
            anchor = observer.content_intervals(
                data["system"], r["user"], [r["anchor_span"]]
            )
            if len(anchor) != 1 or any(
                len(observer.candidate_ids(ids, a)) != 1 for a in ("A", "B")
            ):
                raise ValueError("Frozen one-token anchor/candidates required")
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "anchor": anchor,
                "partitions": partitions(len(ids), span),
            }
        for r in rows:
            l = layouts[r["case_id"]]
            d = layouts[
                lookup[
                    r["group_id"],
                    -r["level"],
                    r["entity"],
                    r["criterion"],
                    r["record_order"],
                ]
            ]
            check_donor_layout(l["ids"], l["span"], d["ids"], d["span"])
            other = layouts[
                lookup[
                    r["group_id"],
                    r["level"],
                    r["entity"],
                    r["criterion"],
                    "reversed" if r["record_order"] == "original" else "original",
                ]
            ]
            if len(l["ids"]) != len(other["ids"]) or l["span"] != other["span"]:
                raise ValueError(
                    "Reversal changes token layout; no positional repair allowed"
                )
        write_json(out / "token_layouts.json", layouts)

        def budget():
            """Apply a finite resource cap before every forward."""
            if (
                observer.calls >= 400
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1M resource cap")

        def record(r, logits):
            """Persist raw A/B logits with role and expected fact orientation."""
            a, b = [
                observer.candidate_ids(layouts[r["case_id"]]["ids"], v)[0]
                for v in ("A", "B")
            ]
            return {
                k: r[k]
                for k in (
                    *keys,
                    "case_id",
                    "parent_case",
                    "target_kind",
                    "role_sign",
                    "expected_sign",
                    "last_named",
                )
            } | {
                "margin": float(logits[a]) - float(logits[b]),
                "high_logit": float(logits[a]),
                "low_logit": float(logits[b]),
            }

        for r in rows:
            budget()
            ids = layouts[r["case_id"]]["ids"]
            captured, logits = observer.capture(ids, positions=list(range(len(ids))))
            states[r["case_id"]] = captured["block_16"]
            np.savez_compressed(
                out / "activations" / (r["case_id"] + ".npz"),
                block_16=captured["block_16"],
            )
            base.append(record(r, logits))
        checks = []
        for r in rows:
            case = r["case_id"]
            source = lookup[
                r["group_id"],
                r["level"],
                r["entity"],
                "event" if r["target_kind"] == "person" else "venue_object",
                r["record_order"],
            ]
            anchor = layouts[case]["anchor"]
            if (
                anchor != layouts[source]["anchor"]
                or layouts[case]["ids"][: anchor[-1] + 1]
                != layouts[source]["ids"][: anchor[-1] + 1]
            ):
                raise ValueError("Noncommon anchor prefix")
            error = float(abs(states[case][anchor] - states[source][anchor]).max())
            checks.append(
                {"case_id": case, "source_case": source, "state_error": error}
            )
        common = max(r["state_error"] for r in checks)
        pd.DataFrame(checks).to_csv(out / "prefix_checks.csv", index=False)
        old = pd.read_csv(args.parent_dir / "baselines.csv").set_index("case_id").margin
        canonical = pd.DataFrame(base).query(
            "record_order == 'original' and target_kind == 'person'"
        )
        replay = float(abs(canonical.margin - canonical.parent_case.map(old)).max())
        if max(common, replay) > 1e-5 or len(canonical) != 48:
            raise ValueError("Common state/parent replay mismatch")
        flush("NATIVE_COMPLETE", prefix_state_max=common, parent_replay_max=replay)
        checked = set()
        for i, r in enumerate(rows):
            case, l = r["case_id"], layouts[r["case_id"]]
            donor = lookup[
                r["group_id"],
                -r["level"],
                r["entity"],
                r["criterion"],
                r["record_order"],
            ]
            shared = lookup[
                r["group_id"],
                -r["level"],
                r["entity"],
                "event" if r["target_kind"] == "person" else "venue_object",
                r["record_order"],
            ]
            if r["target_kind"] not in checked:
                checked.add(r["target_kind"])
                budget()
                _, same = observer.capture(
                    l["ids"],
                    patch=Patch("block_16", l["anchor"], states[case][l["anchor"]]),
                )
                budget()
                _, restore = observer.capture(l["ids"])
                error = max(
                    abs(record(r, z)["margin"] - base[i]["margin"])
                    for z in (same, restore)
                )
                controls.append({"case_id": case, "self_restore_max": error})
                if error > 1e-5:
                    raise ValueError("Self/restore failed")
            for patch, indices in {
                "anchor": l["anchor"],
                "EQD": sorted(j for part in l["partitions"].values() for j in part),
            }.items():
                budget()
                source = shared if patch == "anchor" else donor
                _, logits = observer.capture(
                    l["ids"], patch=Patch("block_16", indices, states[source][indices])
                )
                effects.append(
                    record(r, logits)
                    | {"patch": patch, "donor_case": donor, "state_source_case": source}
                )
            if (i + 1) % 28 == 0:
                flush(
                    "PATCH_RUNNING", prefix_state_max=common, parent_replay_max=replay
                )
                print(f"P1M {i + 1}/112 calls={observer.calls}", flush=True)
        tables = summarize(pd.DataFrame(base), pd.DataFrame(effects))
        restore = float(
            tables["response_rows"].query("patch == 'EQD'").restore_error.abs().max()
        )
        if observer.calls != 340 or len(effects) != 224 or restore > 1e-5:
            raise ValueError("Packet incomplete/full restore failed")
        for name, table in tables.items():
            table.to_csv(out / (name + ".csv"), index=False)
        result = flush(
            "COMPLETE",
            prefix_state_max=common,
            parent_replay_max=replay,
            full_restore_max=restore,
        )
        write_json(
            out / "output_hashes.json",
            {
                str(p.relative_to(out)): file_hash(p)
                for p in out.rglob("*")
                if p.is_file() and not p.name.startswith("._")
            },
        )
        print(json.dumps(result), flush=True)
    except Exception as exc:
        flush(
            "PARTIAL_BUDGET" if isinstance(exc, BudgetExceeded) else "FAILED",
            error=str(exc),
        )
        (out / "error_log.txt").write_text(traceback.format_exc())
        raise


def main():
    """Run one authored-material packet offline without fitting or new labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()

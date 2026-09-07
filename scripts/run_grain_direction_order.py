#!/usr/bin/env python3
"""P1R: paired record-order reversal with frozen attribute directions."""

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
from scripts.run_grain_attribute_direction import components, summarize
from scripts.run_grain_evidence_conduit import partitions
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_vector_mechanism_controls import write_json


def reverse_material(parent):
    """Reverse only two complete named action lines; keep role truth and questions."""
    rows = []
    for old in parent:
        if old["entity"] == "room":
            continue
        lines = old["evidence"].splitlines()
        if len(lines) != 3 or {x.split()[0] for x in lines[1:]} != {"Alex", "Blair"}:
            raise ValueError("Expected exactly two named action lines")
        evidence = "\n".join([lines[0], lines[2], lines[1]])
        if len(evidence) != len(old["evidence"]):
            raise ValueError("Character length changed")
        rows.append(
            {
                **old,
                "case_id": old["case_id"] + "__reversed",
                "parent_case": old["case_id"],
                "record_order": "reversed",
                "evidence": evidence,
                "user": evidence + old["user"][len(old["evidence"]) :],
            }
        )
    if len(rows) != 64 or len({r["user"] for r in rows}) != 64:
        raise ValueError("Expected64 complete person cells")
    return rows


def compare_orders(current, old):
    """Make paired differences without choosing favorable criteria or settings."""
    keys = ["scope", "group_id", "criterion", "axis", "patch"]
    old = old[
        (old.criterion != "venue_object")
        & (old.axis == "B")
        & old.patch.isin(["natural", "projected", "complement"])
    ].copy()
    a, b = current.copy().fillna({"group_id": "ALL"}), old.fillna({"group_id": "ALL"})
    pairs = b.merge(
        a,
        on=keys,
        how="outer",
        suffixes=("_original", "_reversed"),
        validate="one_to_one",
        indicator=True,
    )
    if len(pairs) != 30 or not (pairs._merge == "both").all():
        raise ValueError("Order comparison population")
    for metric in (
        "natural_rms",
        "shift_rms",
        "error_rms",
        "relative_rms",
        "relative_error",
        "intended_fraction",
    ):
        pairs[metric + "_reversed_minus_original"] = (
            pairs[metric + "_reversed"] - pairs[metric + "_original"]
        )
    return pairs.drop(columns="_merge")


def run(args):
    """Run the single declared order panel with raw-logit instrumentation receipts."""
    parent = args.parent_dir
    if json.loads((parent / "run_status.json").read_text())["status"] != "COMPLETE":
        raise ValueError("Complete P1P required")
    for name, digest in json.loads((parent / "output_hashes.json").read_text()).items():
        if file_hash(parent / name) != digest:
            raise ValueError("Parent changed")
    data = json.loads((parent / "materials.json").read_text())
    rows = reverse_material(data["rows"])
    old_manifest = json.loads((parent / "manifest.json").read_text())
    directions = np.load(parent / "directions.npz")
    old_base = pd.read_csv(parent / "baselines.csv").set_index("case_id")
    old_layouts = json.loads((parent / "token_layouts.json").read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("activations", "patch_states"):
        (out / name).mkdir()
    sources = list(
        dict.fromkeys(
            [
                Path(__file__),
                ROOT / "docs/GRAIN_DIRECTION_ORDER_P1R_PLAN.md",
                ROOT / "tests/test_grain_direction_order.py",
            ]
            + [ROOT / p for p in old_manifest["code"]]
        )
    )
    manifest = {
        "parent_identity": old_manifest["identity"],
        "direction_sha256": file_hash(parent / "directions.npz"),
        "material_hash": material_hash(rows),
        "max_calls": 300,
        "max_seconds": 300,
        "private_data_read": False,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": data["system"], "rows": rows})
    (out / "directions.npz").write_bytes((parent / "directions.npz").read_bytes())
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    base, effects, checks, geometry = [], [], [], []
    observer, start = None, time.monotonic()

    def flush(status, **extra):
        """Preserve complete settings and any partial failure without promoting it."""
        for name, values in (
            ("baselines", base),
            ("effects", effects),
            ("geometry", geometry),
            ("instrument_checks", checks),
        ):
            pd.DataFrame(values).to_csv(out / (name + ".csv"), index=False)
        receipt = {
            "status": status,
            "identity": manifest["identity"],
            "calls": observer.calls if observer else 0,
            "seconds": time.monotonic() - start,
            "baselines": len(base),
            "effects": len(effects),
            **extra,
        }
        write_json(out / "run_status.json", receipt)
        return receipt

    try:
        expected = EXPECTED_METADATA | {
            "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        }
        if {k: file_hash(args.model_dir / k) for k in expected} != expected:
            raise ValueError("Changed model")
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
        keys = ("group_id", "level", "room_sign", "entity", "criterion")
        lookup = {tuple(r[k] for k in keys): r["case_id"] for r in rows}
        layouts, donors, states = {}, {}, {}
        for r in rows:
            ids = observer.prompt_ids(data["system"], r["user"])
            span = observer.content_indices(data["system"], r["user"], r["evidence"])
            n = observer.content_intervals(
                data["system"], r["user"], [r["anchor_span"]]
            )
            if len(n) != 1 or any(
                len(observer.candidate_ids(ids, v)) != 1 for v in ("A", "B")
            ):
                raise ValueError("Single-token target/answers required")
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "anchor": n,
                "partitions": partitions(len(ids), span),
            }
            donors[r["case_id"]] = {
                "full": lookup[
                    r["group_id"],
                    -r["level"],
                    r["room_sign"],
                    r["entity"],
                    r["criterion"],
                ],
                "anchor": lookup[
                    r["group_id"], -r["level"], r["room_sign"], r["entity"], "event"
                ],
            }
        for case, donor in donors.items():
            l, d = layouts[case], layouts[donor["full"]]
            check_donor_layout(l["ids"], l["span"], d["ids"], d["span"])
            if layouts[donor["anchor"]]["anchor"] != l["anchor"]:
                raise ValueError("Misaligned anchor")
        write_json(out / "token_layouts.json", layouts)
        write_json(out / "donors.json", donors)

        def budget():
            """Stop on frozen resource limits, not on favorable intermediate outcomes."""
            if (
                observer.calls >= 300
                or time.monotonic() - start > 300
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1R cap")

        def score(ids, logits):
            """Keep raw candidate logits so control differences can be replayed."""
            a, b = [observer.candidate_ids(ids, v)[0] for v in ("A", "B")]
            return {
                "high_logit": float(logits[a]),
                "low_logit": float(logits[b]),
                "margin": float(logits[a]) - float(logits[b]),
            }

        def record(r, logits):
            """Record unchanged material metadata next to candidate scores."""
            return {
                k: r[k]
                for k in (
                    *keys,
                    "case_id",
                    "parent_case",
                    "role_sign",
                    "expected_sign",
                    "action_family",
                )
            } | score(layouts[r["case_id"]]["ids"], logits)

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
        baseline = {r["case_id"]: r["margin"] for r in base}
        prefix = 0.0
        for r in rows:
            source = lookup[
                r["group_id"], r["level"], r["room_sign"], r["entity"], "event"
            ]
            n = layouts[r["case_id"]]["anchor"]
            if (
                layouts[source]["ids"][: n[-1] + 1]
                != layouts[r["case_id"]]["ids"][: n[-1] + 1]
            ):
                raise ValueError("Common prefix changed")
            prefix = max(
                prefix, float(abs(states[source][n] - states[r["case_id"]][n]).max())
            )
        if prefix > 1e-5:
            raise ValueError("Common state failed")
        flush("NATIVE_COMPLETE", prefix_state_max=prefix)
        checked_names, checked_queries = set(), set()
        for i, r in enumerate(rows):
            case, l = r["case_id"], layouts[r["case_id"]]
            n = l["anchor"]
            pending = []
            if r["entity"] not in checked_names:
                checked_names.add(r["entity"])
                pending.extend(
                    [
                        (
                            "identity",
                            l["ids"],
                            Patch("block_16", n, states[case][n]),
                            case,
                            baseline[case],
                            n,
                        ),
                        ("restore", l["ids"], None, case, baseline[case], []),
                    ]
                )
            key = r["entity"], r["criterion"]
            if key not in checked_queries:
                checked_queries.add(key)
                source = donors[case]["full"]
                positions = sorted(j for part in l["partitions"].values() for j in part)
                pending.extend(
                    [
                        (
                            "full",
                            l["ids"],
                            Patch("block_16", positions, states[source][positions]),
                            source,
                            baseline[source],
                            positions,
                        ),
                        (
                            "original_replay",
                            old_layouts[r["parent_case"]]["ids"],
                            None,
                            r["parent_case"],
                            old_base.loc[r["parent_case"], "margin"],
                            [],
                        ),
                    ]
                )
            for kind, ids, patch, source, expected_score, positions in pending:
                budget()
                _, logits = observer.capture(ids, patch=patch)
                value = score(ids, logits)
                error = abs(value["margin"] - expected_score)
                checks.append(
                    {
                        "case_id": case,
                        "check": kind,
                        "source_case": source,
                        "source_set": "parent"
                        if kind == "original_replay"
                        else "current",
                        "positions": json.dumps(positions),
                        **value,
                        "expected_margin": expected_score,
                        "error": error,
                    }
                )
                if error > 1e-5:
                    raise ValueError("Instrument replay failed")
            source = donors[case]["anchor"]
            x, y = states[case][n][0], states[source][n][0]
            d, q = (
                directions[r["entity"] + "_direction"],
                directions[r["entity"] + "_random"],
            )
            values = components(x, y, d, q)
            saved = {
                k: values[k].astype(np.float32)
                for k in ("natural", "projected", "complement")
            }
            np.savez_compressed(out / "patch_states" / (case + ".npz"), **saved)
            delta = y.astype(float) - x.astype(float)
            geometry.append(
                {
                    "case_id": case,
                    "parent_case": r["parent_case"],
                    "delta_norm": float(np.linalg.norm(delta)),
                    "projection_energy_fraction": float(
                        np.dot(d, delta) ** 2 / np.dot(delta, delta)
                    ),
                    "fp32_cast_error": max(
                        float(abs(saved[k].astype(float) - values[k]).max())
                        for k in saved
                    ),
                }
            )
            for kind, vector in saved.items():
                budget()
                _, logits = observer.capture(
                    l["ids"], patch=Patch("block_16", n, vector[None, :])
                )
                effects.append(
                    record(r, logits)
                    | {"axis": "B", "patch": kind, "source_case": source}
                )
            if (i + 1) % 16 == 0:
                print(
                    json.dumps(flush("PATCHING", completed_settings=(i + 1) // 16)),
                    flush=True,
                )
        if observer.calls != 268 or len(effects) != 192 or len(checks) != 12:
            raise ValueError("Incomplete declared panel")
        response, summary = summarize(pd.DataFrame(base), pd.DataFrame(effects))
        pairs = compare_orders(summary, pd.read_csv(parent / "effect_summary.csv"))
        for name, frame in (
            ("response_rows", response),
            ("effect_summary", summary),
            ("order_comparison", pairs),
        ):
            frame.to_csv(out / (name + ".csv"), index=False)
        status = flush(
            "COMPLETE",
            prefix_state_max=prefix,
            instrument_max=max(c["error"] for c in checks),
        )
        write_json(
            out / "output_hashes.json",
            {
                str(p.relative_to(out)): file_hash(p)
                for p in out.rglob("*")
                if p.is_file() and not p.name.startswith("._")
            },
        )
        print(json.dumps(status))
    except Exception as exc:
        flush(
            "PARTIAL_BUDGET" if isinstance(exc, BudgetExceeded) else "FAILED",
            error=str(exc),
        )
        (out / "error_log.txt").write_text(traceback.format_exc())
        raise


def main():
    """Run one immutable panel from explicit local paths."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-dir", "model-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()

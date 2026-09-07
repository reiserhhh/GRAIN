#!/usr/bin/env python3
"""Common entity-prefix experiment with criterion-specific readouts and both people."""

from __future__ import annotations

import argparse
import json
import os
import re
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


def material(parent_rows):
    """Use both entities from every event, with a literal criterion-free prefix."""
    rows = []
    criteria = {
        "event": "did this person initiate the conversation rather than wait?",
        "trait": "does this person's behavior seem outgoing rather than reserved?",
        "venue": "was the room open during the event?",
    }
    for original in parent_rows:
        if (
            original["variant"] != "event_target"
            or original["form"] != "question_first"
        ):
            continue
        primary = re.search(r"Did (Alex|Blair)\b", original["clauses"]["C"]).group(1)
        for entity in ("Alex", "Blair"):
            prefix = original["evidence"] + "\n\nQuestion: For"
            start = len(prefix)
            prefix += " " + entity
            span = [start, len(prefix)]
            role_sign = original["level"] * (1 if entity == primary else -1)
            for criterion, question in criteria.items():
                rows.append(
                    {
                        "case_id": f"{original['group_id']}__{original['level']}__{entity}__{criterion}",
                        "group_id": original["group_id"],
                        "level": original["level"],
                        "entity": entity,
                        "criterion": criterion,
                        "expected_sign": 1 if criterion == "venue" else role_sign,
                        "role_sign": role_sign,
                        "evidence": original["evidence"],
                        "entity_span": span,
                        "user": prefix
                        + ", "
                        + question
                        + " Return A for Yes and B for No.",
                        "parent_case": original["case_id"],
                    }
                )
    if len(rows) != 48 or len({r["case_id"] for r in rows}) != 48:
        raise ValueError("Expected the complete48-prompt panel")
    return rows


def verify_common_prefix(rows, layouts, states):
    """Verify future-criterion independence at actual causal-prefix positions."""
    checks = []
    for key, group in pd.DataFrame(rows).groupby(["group_id", "level", "entity"]):
        if set(group.criterion) != {"event", "trait", "venue"}:
            raise ValueError("Missing criterion")
        source = group[group.criterion == "event"].case_id.iloc[0]
        region = layouts[source]["N"]
        prefix = layouts[source]["ids"][: max(region) + 1]
        for case in group.case_id:
            if (
                layouts[case]["N"] != region
                or layouts[case]["ids"][: max(region) + 1] != prefix
            ):
                raise ValueError("Prefix changes before entity endpoint")
            error = float(np.max(abs(states[case][region] - states[source][region])))
            if error > 1e-5:
                raise ValueError(
                    "Common prefix state differs beyond declared tolerance"
                )
            checks.append(
                {
                    "group_id": key[0],
                    "level": int(key[1]),
                    "entity": key[2],
                    "case_id": case,
                    "source_case": source,
                    "max_state_difference": error,
                }
            )
    return checks


def summarize(base, effects):
    """Keep intended behavior and actual model restoration as distinct outcomes."""
    lookup = base.set_index("case_id").margin
    rows = effects.copy()
    rows["shift"] = rows.margin - rows.case_id.map(lookup)
    rows["gap"] = rows.donor_case.map(lookup) - rows.case_id.map(lookup)
    rows["restore_error"] = rows.margin - rows.donor_case.map(lookup)
    summaries, native, pairs = [], [], []
    for criterion, sub in base.groupby("criterion"):
        for (setting, entity), pair in sub.groupby(["group_id", "entity"]):
            pair = pair.set_index("role_sign")
            difference = float(pair.loc[1, "margin"] - pair.loc[-1, "margin"])
            pairs.append(
                {
                    "criterion": criterion,
                    "group_id": setting,
                    "entity": entity,
                    "role_aligned_gap": difference,
                }
            )
        selected = [r["role_aligned_gap"] for r in pairs if r["criterion"] == criterion]
        native.append(
            {
                "criterion": criterion,
                "pairs": len(selected),
                "positive_role_pairs": int(np.sum(np.array(selected) > 0)),
                "mean_role_gap": float(np.mean(selected)),
                "expected_preference": float(
                    np.mean(sub.margin.to_numpy() * sub.expected_sign.to_numpy() > 0)
                ),
            }
        )
    for (criterion, patch), sub in rows.groupby(["criterion", "patch"]):
        energy = float(np.sum(sub.gap**2))
        summaries.append(
            {
                "criterion": criterion,
                "patch": patch,
                "cases": len(sub),
                "relative_error_energy": float(np.sum(sub.restore_error**2) / energy)
                if energy > 1e-12
                else None,
                "gap_rms": float(np.sqrt(np.mean(sub.gap**2))),
                "shift_rms": float(np.sqrt(np.mean(sub["shift"] ** 2))),
                "toward_donor_fraction": float(np.mean(sub["shift"] * sub.gap > 0)),
                "intended_behavior_fraction": float(
                    np.mean(sub["shift"] * -sub.role_sign > 0)
                )
                if criterion != "venue"
                else None,
                "intended_effect_mean": float(np.mean(sub["shift"] * -sub.role_sign))
                if criterion != "venue"
                else None,
            }
        )
    return {
        "response_rows": rows,
        "transport_summary": pd.DataFrame(summaries),
        "native_summary": pd.DataFrame(native),
        "native_role_pairs": pd.DataFrame(pairs),
    }


def run(args):
    """Execute a complete packet with frozen shape, model and scoring."""
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    if (
        json.loads((args.parent_dir / "run_status.json").read_text())["status"]
        != "COMPLETE"
    ):
        raise ValueError("Complete parent required")
    for name, expected in json.loads(
        (args.parent_dir / "output_hashes.json").read_text()
    ).items():
        if file_hash(args.parent_dir / name) != expected:
            raise ValueError("Parent artifact changed")
    original = json.loads((args.parent_dir / "materials.json").read_text())
    rows = material(original["rows"])
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "activations").mkdir()
    sources = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/multicomment.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "grain_research/vector_mechanisms/role_opportunity.py",
        ROOT / "scripts/run_grain_evidence_conduit.py",
        ROOT / "scripts/run_grain_persona_reciprocity.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "docs/GRAIN_ENTITY_STATE_P1J_PLAN.md",
        ROOT / "tests/test_grain_entity_state.py",
    ]
    manifest = {
        "parent_identity": parent["identity"],
        "material_hash": material_hash(rows),
        "max_calls": 300,
        "max_seconds": 600,
        "pad_to_length": 256,
        "private_data_read": False,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": original["system"], "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    observer, start = None, time.monotonic()
    base, effects, controls = [], [], []
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )

    def flush(status, **extra):
        """Preserve complete or partial raw rows and actual resource usage."""
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
        layouts, states = {}, {}
        lookup = {
            (r["group_id"], r["level"], r["entity"], r["criterion"]): r["case_id"]
            for r in rows
        }
        for r in rows:
            ids = observer.prompt_ids(original["system"], r["user"])
            span = observer.content_indices(
                original["system"], r["user"], r["evidence"]
            )
            name = observer.content_intervals(
                original["system"], r["user"], [r["entity_span"]]
            )
            if len(name) != 1 or any(
                len(observer.candidate_ids(ids, a)) != 1 for a in ("A", "B")
            ):
                raise ValueError("Frozen one-token name/candidates required")
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "N": name,
                "partitions": partitions(len(ids), span),
            }
        for r in rows:
            layout = layouts[r["case_id"]]
            donor = layouts[
                lookup[r["group_id"], -r["level"], r["entity"], r["criterion"]]
            ]
            check_donor_layout(
                layout["ids"], layout["span"], donor["ids"], donor["span"]
            )
        write_json(out / "token_layouts.json", layouts)

        def budget(n=1):
            """Enforce finite complete-case work, not unlimited diagnostic retries."""
            if (
                observer.calls + n > 300
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1J resource cap")

        def record(r, logits):
            """Save common-code candidates with intended subject orientation separate."""
            ids = layouts[r["case_id"]]["ids"]
            hi, lo = [observer.candidate_ids(ids, a)[0] for a in ("A", "B")]
            return {
                k: r[k]
                for k in (
                    "case_id",
                    "group_id",
                    "level",
                    "entity",
                    "criterion",
                    "role_sign",
                    "expected_sign",
                )
            } | {
                "margin": float(logits[hi]) - float(logits[lo]),
                "high_logit": float(logits[hi]),
                "low_logit": float(logits[lo]),
            }

        for r in rows:
            budget()
            case = r["case_id"]
            ids = layouts[case]["ids"]
            captured, logits = observer.capture(ids, positions=list(range(len(ids))))
            states[case] = captured["block_16"]
            np.savez_compressed(
                out / "activations" / (case + ".npz"), block_16=states[case]
            )
            base.append(record(r, logits))
        shared = verify_common_prefix(rows, layouts, states)
        pd.DataFrame(shared).to_csv(out / "shared_prefix_checks.csv", index=False)
        flush(
            "NATIVE_COMPLETE",
            common_prefix_error=max(r["max_state_difference"] for r in shared),
        )
        for i, r in enumerate(rows):
            budget(5)
            case = r["case_id"]
            layout = layouts[case]
            donor = lookup[r["group_id"], -r["level"], r["entity"], r["criterion"]]
            shared_donor = lookup[r["group_id"], -r["level"], r["entity"], "event"]
            if i in (0, 3):
                _, same = observer.capture(
                    layout["ids"],
                    patch=Patch("block_16", layout["N"], states[case][layout["N"]]),
                )
                _, restored = observer.capture(layout["ids"])
                err = max(
                    abs(record(r, same)["margin"] - base[i]["margin"]),
                    abs(record(r, restored)["margin"] - base[i]["margin"]),
                )
                controls.append({"case_id": case, "self_restore_max": err})
                if err > 1e-5:
                    raise ValueError("Self/restore mismatch")
            parts = {
                "N": layout["N"],
                "Q": layout["partitions"]["Q"],
                "EQD": sorted(j for p in layout["partitions"].values() for j in p),
            }
            for patch, indices in parts.items():
                budget()
                source = shared_donor if patch == "N" else donor
                _, logits = observer.capture(
                    layout["ids"],
                    patch=Patch("block_16", indices, states[source][indices]),
                )
                effects.append(
                    record(r, logits)
                    | {"patch": patch, "donor_case": donor, "state_source_case": source}
                )
            if (i + 1) % 12 == 0:
                flush("PATCH_RUNNING")
                print(f"P1J {i + 1}/48 calls={observer.calls}", flush=True)
        tables = summarize(pd.DataFrame(base), pd.DataFrame(effects))
        error = float(
            tables["response_rows"].query("patch == 'EQD'").restore_error.abs().max()
        )
        if observer.calls != 196 or len(effects) != 144 or error > 1e-5:
            raise ValueError("Full packet/restoration failed")
        for name, table in tables.items():
            table.to_csv(out / (name + ".csv"), index=False)
        result = flush(
            "COMPLETE",
            full_donor_restore_max=error,
            common_prefix_error=max(r["max_state_difference"] for r in shared),
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
    """Run one offline packet with no training, external calls or private data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Separate semantic query roles from relative suffix position using clause order."""

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
from grain_research.vector_mechanisms.role_opportunity import SYSTEM, check_donor_layout
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_evidence_conduit import partitions
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_query_transport_transfer import material as prior_material
from scripts.run_grain_vector_mechanism_controls import write_json


def material():
    """Keep the exact canonical clauses, moving only their order in48 synthetic prompts."""
    rows = []
    for original in prior_material():
        if original["form"] != "canonical":
            continue
        prefix = original["evidence"] + "\n\nQuestion:"
        convention = (
            "Answer Yes or No."
            if original["variant"] == "event_target"
            else "Return A for Yes and B for No."
        )
        original_question = original["user"][len(prefix) :].strip()
        if not original_question.endswith(" " + convention):
            raise ValueError("Canonical convention changed")
        content = original_question[: -(len(convention) + 1)]
        clauses = {"C": content, "A": convention}
        for order, keys in (
            ("question_first", ("C", "A")),
            ("convention_first", ("A", "C")),
        ):
            user, intervals = prefix, {}
            for key in keys:
                start = len(user)
                user += " " + clauses[key]
                intervals[key] = [start, len(user)]
            row = original | {
                "form": order,
                "user": user,
                "intervals": intervals,
                "clauses": clauses,
                "parent_case": original["case_id"],
                "case_id": original["case_id"].replace("canonical", order),
            }
            if order == "question_first" and user != original["user"]:
                raise ValueError("Canonical replay changed")
            rows.append(row)
    return rows


def semantic_partition(q_indices, content, answer):
    """Make exhaustive disjoint Q subregions with every boundary token accounted for."""
    q, c, a = map(set, (q_indices, content, answer))
    if not c or not a or c & a or not (c | a) <= q:
        raise ValueError("Semantic query intervals overlap or exceed Q")
    r = q - c - a
    if not r:
        raise ValueError("No remainder interval")
    return {"C": sorted(c), "A": sorted(a), "R": sorted(r)}


def summarize(base, effects):
    """Keep raw scores and matched-setting effects without a personality-validity claim."""
    lookup = base.set_index("case_id").margin
    effects = effects.copy()
    effects["shift"] = effects.margin - effects.case_id.map(lookup)
    effects["gap"] = effects.donor_case.map(lookup) - effects.case_id.map(lookup)
    effects["error"] = effects.margin - effects.donor_case.map(lookup)
    transport, native, differences = [], [], []
    for (variant, form, patch), sub in effects.groupby(["variant", "form", "patch"]):
        denom = float(np.dot(sub.gap, sub.gap))
        transport.append(
            {
                "variant": variant,
                "form": form,
                "patch": patch,
                "cases": len(sub),
                "gap_rms": float(np.sqrt(np.mean(sub.gap**2))),
                "relative_error_energy": float(np.dot(sub.error, sub.error) / denom)
                if denom > 1e-12
                else None,
                "shift_rms": float(np.sqrt(np.mean(sub["shift"] ** 2))),
                "mean_absolute_shift": float(sub["shift"].abs().mean()),
            }
        )
    for (variant, form), sub in base.groupby(["variant", "form"]):
        pairs = sub.pivot(index="group_id", columns="level", values="margin")
        gap = pairs[1] - pairs[-1]
        sign = np.ones(len(sub)) if variant == "venue" else sub.level.to_numpy()
        native.append(
            {
                "variant": variant,
                "form": form,
                "positive_pairs": int((gap > 0).sum()),
                "pairs": len(gap),
                "gap_mean": float(gap.mean()),
                "absolute_expected_preference": float(
                    np.mean(sub.margin.to_numpy() * sign > 0)
                ),
            }
        )
    for (variant, group, level), sub in effects.groupby(
        ["variant", "group_id", "level"]
    ):
        sub = sub.set_index(["form", "patch"])
        for form in ("question_first", "convention_first"):

            def shift(patch, sub=sub, form=form):
                """Read a declared region within this matched recipient."""
                return float(sub.loc[(form, patch), "shift"])

            early, late = ("C", "A") if form == "question_first" else ("A", "C")
            differences.append(
                {
                    "variant": variant,
                    "group_id": group,
                    "level": level,
                    "form": form,
                    "content_shift": shift("C"),
                    "convention_shift": shift("A"),
                    "remainder_shift": shift("R"),
                    "early_shift": shift(early),
                    "late_shift": shift(late),
                    "all_query_shift": shift("Q"),
                    "additivity_defect": shift("Q")
                    - shift("C")
                    - shift("A")
                    - shift("R"),
                }
            )
    return {
        "transport_rows": effects,
        "transport_summary": pd.DataFrame(transport),
        "native_summary": pd.DataFrame(native),
        "matched_role_position": pd.DataFrame(differences),
    }


def run(args):
    """Collect one bounded synthetic packet without changing any predecessor."""
    parent = json.loads((args.previous_dir / "manifest.json").read_text())
    if (
        json.loads((args.previous_dir / "run_status.json").read_text())["status"]
        != "COMPLETE"
    ):
        raise ValueError("Complete parent required")
    for name, expected in json.loads(
        (args.previous_dir / "output_hashes.json").read_text()
    ).items():
        if file_hash(args.previous_dir / name) != expected:
            raise ValueError("Parent changed")
    rows = material()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "activations").mkdir()
    sources = [
        Path(__file__),
        ROOT / "scripts/run_grain_query_transport_transfer.py",
        ROOT / "scripts/run_grain_evidence_conduit.py",
        ROOT / "scripts/run_grain_persona_reciprocity.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/multicomment.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "grain_research/vector_mechanisms/role_opportunity.py",
        ROOT / "docs/GRAIN_QUERY_COORDINATE_P1H_PLAN.md",
        ROOT / "tests/test_grain_query_coordinate.py",
    ]
    manifest = {
        "parent_identity": parent["identity"],
        "material_hash": material_hash(rows),
        "max_calls": 500,
        "max_seconds": 600,
        "private_data_read": False,
        "site": "block_16",
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": SYSTEM, "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    observer, start = None, time.monotonic()
    base, effects, checks = [], [], []
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )

    def flush(status, **extra):
        """Preserve partial results and resource receipts without declaring completion."""
        pd.DataFrame(base).to_csv(out / "baselines.csv", index=False)
        pd.DataFrame(effects).to_csv(out / "effects.csv", index=False)
        write_json(out / "checks.json", checks)
        state = {
            "status": status,
            "identity": manifest["identity"],
            "calls": observer.calls if observer else 0,
            "seconds": time.monotonic() - start,
            "baselines": len(base),
            "effects": len(effects),
            **extra,
        }
        write_json(out / "run_status.json", state)
        return state

    try:
        expected = EXPECTED_METADATA | {
            "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        }
        if {k: file_hash(args.model_dir / k) for k in expected} != expected:
            raise ValueError("Model changed")
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(args.model_dir), device="mps", dtype="float32", max_tokens=512
        )
        observer.sites = {"block_16": observer.model.model.layers[16]}
        write_json(out / "model_manifest.json", observer.manifest())
        layouts, states = {}, {}
        lookup = {
            (r["group_id"], r["variant"], r["form"], r["level"]): r["case_id"]
            for r in rows
        }
        for r in rows:
            ids = observer.prompt_ids(SYSTEM, r["user"])
            span = observer.content_indices(SYSTEM, r["user"], r["evidence"])
            parts = partitions(len(ids), span)
            semantic = semantic_partition(
                parts["Q"],
                *[
                    observer.content_intervals(SYSTEM, r["user"], [r["intervals"][k]])
                    for k in ("C", "A")
                ],
            )
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "partitions": parts,
                "semantic": semantic,
            }
            if any(
                len(observer.candidate_ids(ids, r[key])) != 1
                for key in ("high_answer", "low_answer")
            ):
                raise ValueError("Single-token candidates required")
        for r in rows:
            layout = layouts[r["case_id"]]
            donor = layouts[lookup[r["group_id"], r["variant"], r["form"], -r["level"]]]
            check_donor_layout(
                layout["ids"], layout["span"], donor["ids"], donor["span"]
            )
            if layout["semantic"] != donor["semantic"]:
                raise ValueError("Donor semantic span changed")
        write_json(out / "token_layouts.json", layouts)

        def budget(n=1):
            """Check actual computation, not only planned prompts."""
            if (
                observer.calls + n > 500
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1H resource limit")

        def record(r, logits):
            """Save the native candidate arithmetic and its original semantics."""
            ids = layouts[r["case_id"]]["ids"]
            hi, lo = [
                observer.candidate_ids(ids, r[k])[0]
                for k in ("high_answer", "low_answer")
            ]
            return {
                k: r[k] for k in ("case_id", "group_id", "variant", "form", "level")
            } | {
                "margin": float(logits[hi]) - float(logits[lo]),
                "high_logit": float(logits[hi]),
                "low_logit": float(logits[lo]),
            }

        previous = (
            pd.read_csv(args.previous_dir / "baselines.csv").set_index("case_id").margin
        )
        for r in rows:
            budget()
            case = r["case_id"]
            ids = layouts[case]["ids"]
            captured, logits = observer.capture(ids, positions=list(range(len(ids))))
            states[case] = captured["block_16"]
            np.savez_compressed(
                out / "activations" / (case + ".npz"), block_16=states[case]
            )
            row = record(r, logits)
            if r["form"] == "question_first":
                err = abs(row["margin"] - previous[r["parent_case"]])
                checks.append({"case_id": case, "parent_replay_error": err})
                if err > 1e-5:
                    raise ValueError("Parent native replay failed")
            base.append(row)
        flush("NATIVE_COMPLETE")
        for i, r in enumerate(rows):
            budget(8)
            case = r["case_id"]
            layout = layouts[case]
            donor = lookup[r["group_id"], r["variant"], r["form"], -r["level"]]
            patch_sets = layout["semantic"] | {
                k: layout["partitions"][k] for k in ("Q", "D")
            }
            patch_sets["EQD"] = sorted(
                j for p in layout["partitions"].values() for j in p
            )
            if i < 2:
                _, same = observer.capture(
                    layout["ids"],
                    patch=Patch(
                        "block_16", patch_sets["Q"], states[case][patch_sets["Q"]]
                    ),
                )
                _, restored = observer.capture(layout["ids"])
                err = max(
                    abs(record(r, same)["margin"] - base[i]["margin"]),
                    abs(record(r, restored)["margin"] - base[i]["margin"]),
                )
                checks.append({"case_id": case, "self_restore_error": err})
                if err > 1e-5:
                    raise ValueError("Patch cleanup failed")
            for name, idx in patch_sets.items():
                budget()
                _, logits = observer.capture(
                    layout["ids"], patch=Patch("block_16", idx, states[donor][idx])
                )
                effects.append(
                    record(r, logits)
                    | {"patch": name, "donor_case": donor, "patched_tokens": len(idx)}
                )
            if (i + 1) % 12 == 0:
                flush("PATCH_RUNNING")
                print(f"P1H {i + 1}/48 calls={observer.calls}", flush=True)
        tables = summarize(pd.DataFrame(base), pd.DataFrame(effects))
        err = float(tables["transport_rows"].query("patch == 'EQD'").error.abs().max())
        if len(effects) != 288 or observer.calls != 340 or err > 1e-5:
            raise ValueError("Complete packet or restoration failed")
        for name, table in tables.items():
            table.to_csv(out / (name + ".csv"), index=False)
        result = flush("COMPLETE", full_donor_restore_error=err)
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
    """Run the single declared offline packet, never restarting previous experiments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous-dir", type=Path, required=True)
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

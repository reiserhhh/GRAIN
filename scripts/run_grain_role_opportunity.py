#!/usr/bin/env python3
"""Run a fresh bounded offline role/opportunity mechanism packet; no private data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zipfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.role_opportunity import (
    SYSTEM,
    build_material,
    check_donor_layout,
    fit_directions,
    summarize,
)
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_vector_mechanism_controls import write_json

SITE = "block_16"
METADATA = [
    "case_id",
    "group_id",
    "split",
    "family",
    "level",
    "question_polarity",
    "code",
    "condition",
    "owner",
    "expected_sign",
]


def run(args):
    """Checkpoint native and signed-pair evidence, with no automatic rescue or resume."""
    if not 1 <= args.max_calls <= 2000 or not 0 < args.max_seconds <= 1200:
        raise ValueError("P1B bounded budget exceeded")
    rows = build_material()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Fresh output directory required")
    selected = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/role_opportunity.py",
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/analysis.py",
        ROOT / "grain_research/vector_mechanisms/reciprocity.py",
        ROOT / "scripts/run_grain_persona_reciprocity.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "docs/GRAIN_ROLE_OPPORTUNITY_P1B_PROTOCOL.md",
        ROOT / "tests/test_grain_role_opportunity.py",
    ]
    expected = EXPECTED_METADATA | {
        "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
    }
    manifest = {
        "protocol": "P1B_role_opportunity_v1",
        "material_hash": material_hash(rows),
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in selected},
        "expected_model_hashes": expected,
        "model_dir": str(args.model_dir.resolve()),
        "site": SITE,
        "direction_position": "evidence_end",
        "dose": 0.5,
        "max_calls": args.max_calls,
        "max_seconds": args.max_seconds,
        "native_only": args.native_only,
        "prepare_only": args.prepare_only,
        "private_data_read": False,
        "training": False,
        "claude_called": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": SYSTEM, "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in selected:
            z.write(path, path.relative_to(ROOT))
    if args.prepare_only:
        write_json(
            out / "run_status.json",
            {"status": "PREPARED_ONLY", "cases": len(rows), "calls": 0},
        )
        print("PREPARED_ONLY cases=156 calls=0", flush=True)
        return
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    start, observer = time.monotonic(), None
    baselines, effects, checks, tokens, states, spans = [], [], [], {}, {}, {}
    cache = out / "activation_cache"
    cache.mkdir()

    def flush(status, **extra):
        """Persist honest partial status and all finished records."""
        pd.DataFrame(baselines).to_csv(out / "baselines.csv", index=False)
        pd.DataFrame(effects).to_csv(out / "interventions.csv", index=False)
        write_json(out / "instrument_checks.json", checks)
        state = dict(
            status=status,
            identity=manifest["identity"],
            baseline_rows=len(baselines),
            intervention_rows=len(effects),
            calls=observer.calls if observer else 0,
            seconds=time.monotonic() - start,
            private_data_read=False,
            **extra,
        )
        write_json(out / "run_status.json", state)
        return state

    try:
        if {k: file_hash(args.model_dir / k) for k in expected} != expected:
            raise ValueError("Pinned checkpoint/tokenizer mismatch")
        if not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable; no automatic device switch")
        torch.manual_seed(731)
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(args.model_dir), device="mps", dtype="float32", max_tokens=512
        )
        observer.sites = {SITE: observer.sites[SITE]}
        write_json(out / "model_manifest.json", observer.manifest())

        def budget(cost=1):
            """Guard actual calls, wall time and allocation before new complete work."""
            if (
                observer.calls + cost > args.max_calls
                or time.monotonic() - start > args.max_seconds
            ):
                raise BudgetExceeded("Actual-call or wall-time budget reached")
            if torch.mps.current_allocated_memory() > 9 * 1024**3:
                raise BudgetExceeded("MPS memory budget reached")

        def score(row, patch=None, logits=None):
            """Share a forward for one-token alternatives; otherwise score full sequences."""
            ids = tokens[row["case_id"]]
            answers = [row["high_answer"], row["low_answer"]]
            a = [observer.candidate_ids(ids, x) for x in answers]
            if all(len(x) == 1 for x in a):
                if logits is None:
                    budget()
                    _, logits = observer.capture(ids, patch=patch)
                selected = np.asarray([logits[x[0]] for x in a], dtype=float)
                maximum = float(np.max(logits))
                logz = maximum + float(
                    np.log(np.exp(logits.astype(float) - maximum).sum())
                )
                values = selected - logz
                return (
                    float(values[0] - values[1]),
                    float(np.exp(values).sum()),
                    values.tolist(),
                )
            budget(2)
            values = [observer.score(ids, answer, patch=patch) for answer in answers]
            return values[0] - values[1], None, values

        # Token-only layout admission precedes all neural collection.
        for row in rows:
            case = row["case_id"]
            tokens[case] = observer.prompt_ids(SYSTEM, row["user"])
            spans[case] = observer.content_indices(SYSTEM, row["user"], row["evidence"])
        lookup = {
            (r["group_id"], r["level"]): r
            for r in rows
            if r["family"] == "trait" and r["question_polarity"] == r["code"] == 1
        }
        for (group, level), row in lookup.items():
            donor = lookup[group, -level]
            check_donor_layout(
                tokens[row["case_id"]],
                spans[row["case_id"]],
                tokens[donor["case_id"]],
                spans[donor["case_id"]],
            )
        write_json(
            out / "token_layouts.json",
            {k: {"ids": v, "span": spans[k]} for k, v in tokens.items()},
        )
        for i, row in enumerate(rows):
            case, ids = row["case_id"], tokens[row["case_id"]]
            budget(3)
            # Duplicate endpoint capture is intentional: final two rows always mean
            # evidence-end and decision, regardless of evidence span length.
            positions = spans[case] + [spans[case][-1], len(ids) - 1]
            captured, logits = observer.capture(ids, positions=positions)
            states[case] = captured[SITE]
            margin, mass, values = score(row, logits=logits)
            if not np.isfinite(margin) or not np.isfinite(states[case]).all():
                raise ValueError("Nonfinite baseline")
            np.savez_compressed(cache / (case + ".npz"), state=states[case])
            baselines.append(
                {k: row[k] for k in METADATA}
                | {
                    "margin": margin,
                    "candidate_mass": mass,
                    "candidate_scores": json.dumps(values),
                    "high_ids": json.dumps(
                        observer.candidate_ids(ids, row["high_answer"])
                    ),
                    "low_ids": json.dumps(
                        observer.candidate_ids(ids, row["low_answer"])
                    ),
                    "prompt_tokens": len(ids),
                    "token_hash": material_hash(ids),
                    "evidence_end": spans[case][-1],
                    "decision": len(ids) - 1,
                }
            )
            if i < 2:
                for name, position, state_index in (
                    ("evidence_end", spans[case][-1], -2),
                    ("decision", len(ids) - 1, -1),
                ):
                    budget(2)
                    _, copied = observer.capture(
                        ids, patch=Patch(SITE, position, states[case][state_index])
                    )
                    _, restored = observer.capture(ids)
                    error = max(
                        float(np.max(abs(copied - logits))),
                        float(np.max(abs(restored - logits))),
                    )
                    checks.append({"case_id": case, "position": name, "error": error})
                    if error > 1e-5:
                        raise ValueError("Self-patch/restore failed")
            if (i + 1) % 12 == 0:
                flush("NATIVE_RUNNING")
                print(f"NATIVE {i + 1}/{len(rows)} calls={observer.calls}", flush=True)
        for name, table in summarize(pd.DataFrame(baselines), pd.DataFrame()).items():
            table.to_csv(out / (name + ".csv"), index=False)
        if args.native_only:
            print(
                json.dumps(flush("NATIVE_COMPLETE", full_interventions_run=False)),
                flush=True,
            )
            return
        directions, info = fit_directions(rows, states)
        np.savez_compressed(out / "directions.npz", **directions)
        write_json(out / "direction_metadata.json", info)
        base_lookup = {r["case_id"]: r["margin"] for r in baselines}
        targets = [r for r in rows if r["family"] in ("trait", "venue")]
        for gi, group in enumerate(sorted({r["group_id"] for r in targets})):
            group_rows = [r for r in targets if r["group_id"] == group]
            budget(216)
            for row in group_rows:
                case, ids = row["case_id"], tokens[row["case_id"]]
                position_names = (
                    ("evidence_end", "decision")
                    if row["family"] == "trait"
                    else ("decision",)
                )
                sources = (
                    ("self", "other", "random")
                    if row["family"] == "trait"
                    else ("self", "other")
                )
                for position, source, sign in product(position_names, sources, (-1, 1)):
                    j, index = (
                        (spans[case][-1], -2)
                        if position == "evidence_end"
                        else (len(ids) - 1, -1)
                    )
                    delta = 0.5 * sign * directions[source]
                    margin, mass, values = score(
                        row, Patch(SITE, j, states[case][index].astype(float) + delta)
                    )
                    if not np.isfinite(margin):
                        raise ValueError("Nonfinite intervention")
                    effects.append(
                        {k: row[k] for k in METADATA}
                        | {
                            "kind": "additive",
                            "position": position,
                            "source": source,
                            "sign": sign,
                            "margin": margin,
                            "delta": margin - base_lookup[case],
                            "candidate_mass": mass,
                            "candidate_scores": json.dumps(values),
                            "effect_norm": float(np.linalg.norm(delta)),
                            "relative_norm": float(
                                np.linalg.norm(delta)
                                / np.linalg.norm(states[case][index])
                            ),
                            "donor_case": "",
                        }
                    )
            for level in (-1, 1):
                row = lookup[group, level]
                case = row["case_id"]
                for donor_level in (level, -level):
                    donor_case = lookup[group, donor_level]["case_id"]
                    margin, mass, values = score(
                        row, Patch(SITE, spans[case], states[donor_case][:-2])
                    )
                    effects.append(
                        {k: row[k] for k in METADATA}
                        | {
                            "kind": "donor",
                            "position": "evidence_span",
                            "source": "self_copy"
                            if donor_level == level
                            else "counterfactual",
                            "sign": 0,
                            "margin": margin,
                            "delta": margin - base_lookup[case],
                            "candidate_mass": mass,
                            "candidate_scores": json.dumps(values),
                            "donor_case": donor_case,
                            "effect_norm": float(
                                np.linalg.norm(
                                    states[donor_case][:-2] - states[case][:-2]
                                )
                            ),
                            "relative_norm": None,
                        }
                    )
            flush("INTERVENTIONS_RUNNING", completed_scenarios=gi + 1)
            print(f"INTERVENTIONS {gi + 1}/6 calls={observer.calls}", flush=True)
        for name, table in summarize(
            pd.DataFrame(baselines), pd.DataFrame(effects)
        ).items():
            table.to_csv(out / (name + ".csv"), index=False)
        for row in (targets[0], targets[-1]):
            margin, _, _ = score(row)
            error = abs(margin - base_lookup[row["case_id"]])
            checks.append(
                {"case_id": row["case_id"], "position": "final_restore", "error": error}
            )
            if error > 1e-5:
                raise ValueError("Final restore failed")
        status = flush("COMPLETE", completed_scenarios=6, full_interventions_run=True)
        outputs = [
            p for p in out.iterdir() if p.is_file() and not p.name.startswith("._")
        ]
        write_json(out / "output_hashes.json", {p.name: file_hash(p) for p in outputs})
        print(json.dumps(status), flush=True)
    except Exception as exc:
        flush(
            "PARTIAL_BUDGET" if isinstance(exc, BudgetExceeded) else "FAILED",
            error=str(exc),
        )
        (out / "error_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    """Use existing local FP32 observer with explicit per-packet limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--max-calls", type=int, default=2000)
    parser.add_argument("--max-seconds", type=int, default=1200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

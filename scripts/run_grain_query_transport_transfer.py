#!/usr/bin/env python3
"""Test frozen query-transport formulas on a finite fresh synthetic material panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zipfile
from collections import Counter
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
from scripts.run_grain_query_transport import bin_indices
from scripts.run_grain_vector_mechanism_controls import write_json


def material():
    """Return48 unselected prompts with equal-word-bag opposite ownership assignments."""
    settings = (
        ("f_gardening", "a community gardening demonstration"),
        ("f_astronomy", "an amateur astronomy orientation"),
        ("f_chess", "a chess tournament introduction"),
        ("f_language", "a language exchange reception"),
    )
    actions = {
        1: "greeted someone new, asked their name, and began chatting",
        -1: "stayed silent near someone new until that person began chatting",
    }
    rows = []
    for i, (group, setting) in enumerate(settings):
        target, other = ("Alex", "Blair") if i % 2 == 0 else ("Blair", "Alex")
        for level in (-1, 1):
            statements = [f"{target} {actions[level]}.", f"{other} {actions[-level]}."]
            if i // 2:
                statements.reverse()
            evidence = f"Record of {setting}. The room was open.\n" + "\n".join(
                statements
            )
            queries = {
                "canonical": {
                    "event_target": f"Did {target} initiate the conversation rather than wait? Answer Yes or No.",
                    "trait_code1": f"Based on this record, does {target} seem outgoing rather than reserved? Return A for Yes and B for No.",
                    "venue": "Was the room open? Return A for Yes and B for No.",
                },
                "rewrite": {
                    "event_target": f"Considering what happened, was {target} the one who made the first conversational move? Reply Yes or No.",
                    "trait_code1": f"Consider only {target}'s behavior in the record. Is an outgoing disposition more consistent with it than a reserved one? Reply A for Yes and B for No.",
                    "venue": "According to the record, is it true that the room was open? Reply A for Yes and B for No.",
                },
            }
            for form, family in queries.items():
                for variant, query in family.items():
                    rows.append(
                        {
                            "case_id": f"{group}__{variant}__{form}__{level}",
                            "group_id": group,
                            "variant": variant,
                            "form": form,
                            "level": level,
                            "evidence": evidence,
                            "user": evidence + "\n\nQuestion: " + query,
                            "high_answer": "Yes" if variant == "event_target" else "A",
                            "low_answer": "No" if variant == "event_target" else "B",
                        }
                    )
    for group, _ in settings:
        evidence = {r["level"]: r["evidence"] for r in rows if r["group_id"] == group}
        if Counter(evidence[-1].split()) != Counter(evidence[1].split()):
            raise ValueError("Role pair changed word bag")
    return rows


def summarize(base, effects, layouts, states, shapes, models):
    """Score all frozen models, including failures, without fitting to new outcomes."""
    lookup = base.set_index("case_id").margin
    joined = effects.copy()
    joined["shift"] = joined.margin - joined.case_id.map(lookup)
    joined["donor_gap"] = joined.donor_case.map(lookup) - joined.case_id.map(lookup)
    joined["restore_error"] = joined.margin - joined.donor_case.map(lookup)
    transport, native, fits, predictions = [], [], [], []
    for (variant, form, site, patch), sub in joined.groupby(
        ["variant", "form", "site", "patch"]
    ):
        denom = float(np.sum(sub.donor_gap**2))
        transport.append(
            {
                "variant": variant,
                "form": form,
                "site": site,
                "patch": patch,
                "cases": len(sub),
                "donor_gap_rms": float(np.sqrt(np.mean(sub.donor_gap**2))),
                "relative_error_energy": float(np.sum(sub.restore_error**2) / denom)
                if denom > 1e-12
                else None,
                "mean_absolute_shift": float(sub["shift"].abs().mean()),
            }
        )
    for (variant, form), sub in base.groupby(["variant", "form"]):
        p = sub.pivot(index="group_id", columns="level", values="margin")
        gaps = p[1] - p[-1]
        expected = np.ones(len(sub)) if variant == "venue" else sub.level.to_numpy()
        native.append(
            {
                "variant": variant,
                "form": form,
                "paired_gap_mean": float(gaps.mean()),
                "positive_pairs": int((gaps > 0).sum()),
                "pairs": len(gaps),
                "absolute_expected_preference": float(
                    np.mean(sub.margin.to_numpy() * expected > 0)
                ),
            }
        )
    selected = joined[
        joined.patch.isin(["Q", "bin1", "bin2", "bin3", "bin4"])
        & (joined.site == "block_16")
    ]
    for (variant, form), sub in selected.groupby(["variant", "form"]):
        if variant == "venue":
            continue
        shape = shapes[variant]
        masks = np.array(
            [
                np.ones(4) if r.patch == "Q" else np.eye(4)[int(r.patch[-1]) - 1]
                for r in sub.itertuples()
            ]
        )
        factor = masks @ shape
        for view in ("oracle_gap", "E", "Q", "D"):
            if view == "oracle_gap":
                gaps = sub.donor_gap.to_numpy()
            else:
                beta = models[f"{variant}__{view}"]
                scale = models[f"{variant}__{view}__scale"][0]
                differences = np.stack(
                    [
                        (
                            states[r.donor_case]["block_16"][
                                layouts[r.case_id]["partitions"][view]
                            ]
                            - states[r.case_id]["block_16"][
                                layouts[r.case_id]["partitions"][view]
                            ]
                        )
                        .astype(float)
                        .mean(axis=0)
                        for r in sub.itertuples()
                    ]
                )
                gaps = differences @ beta / scale
            pred = gaps * factor
            observed = sub["shift"].to_numpy()
            fits.append(
                {
                    "variant": variant,
                    "form": form,
                    "gap_view": view,
                    "cases": sub.case_id.nunique(),
                    "settings": sub.group_id.nunique(),
                    "effect_rows": len(sub),
                    "rmse": float(np.sqrt(np.mean((pred - observed) ** 2))),
                    "nochange_rmse": float(np.sqrt(np.mean(observed**2))),
                    "relative_error_energy": float(
                        np.sum((pred - observed) ** 2) / np.sum(observed**2)
                    ),
                    "held_gap_input": view == "oracle_gap",
                    "refitted": False,
                }
            )
            for j, row in enumerate(sub.to_dict("records")):
                predictions.append(
                    row
                    | {
                        "gap_view": view,
                        "predicted_gap": gaps[j],
                        "predicted_shift": pred[j],
                    }
                )
    return {
        "native_summary": pd.DataFrame(native),
        "transport_summary": pd.DataFrame(transport),
        "frozen_operator_scores": pd.DataFrame(fits),
        "frozen_operator_predictions": pd.DataFrame(predictions),
        "transport_rows": joined,
    }


def run(args):
    """Run only this finite synthetic packet, preserving artifacts on interruption."""
    prior = json.loads((args.operator_dir / "summary.json").read_text())
    for name, sha in prior["outputs"].items():
        if file_hash(args.operator_dir / name) != sha:
            raise ValueError("Operator artifact checksum")
    for name, sha in json.loads(
        (args.shape_dir / "output_hashes.json").read_text()
    ).items():
        if file_hash(args.shape_dir / name) != sha:
            raise ValueError("Shape artifact checksum")
    if (
        prior["parent_identity"]
        != json.loads((args.shape_dir / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Parents differ")
    previous = pd.read_csv(args.shape_dir / "operator_scores.csv")
    shapes = {
        r.variant: np.array(json.loads(r.coefficients))
        for r in previous.itertuples()
        if r.degree == 1 and r.split == "fit"
    }
    with np.load(
        args.operator_dir / "linear_gap_models.npz", allow_pickle=False
    ) as data:
        models = {k: data[k] for k in data.files}
    rows = material()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "activations").mkdir()
    sources = [
        Path(__file__),
        ROOT / "scripts/run_grain_query_transport.py",
        ROOT / "scripts/run_grain_evidence_conduit.py",
        ROOT / "scripts/run_grain_persona_reciprocity.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "grain_research/vector_mechanisms/role_opportunity.py",
        ROOT / "docs/GRAIN_QUERY_TRANSPORT_P1F_PROTOCOL.md",
        ROOT / "tests/test_grain_query_transport_transfer.py",
    ]
    manifest = {
        "material_hash": material_hash(rows),
        "parent_identity": prior["identity"],
        "max_calls": 650,
        "max_seconds": 600,
        "private_data_read": False,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": SYSTEM, "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    base, effects, checks = [], [], []
    observer, start = None, time.monotonic()
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )

    def flush(status, **extra):
        """Persist all observations and distinguish complete from interrupted packets."""
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
        observer.sites = {
            f"block_{i:02d}": observer.model.model.layers[i] for i in (4, 16)
        }
        write_json(out / "model_manifest.json", observer.manifest())
        layouts, states = {}, {}
        lookup = {
            (r["group_id"], r["variant"], r["form"], r["level"]): r["case_id"]
            for r in rows
        }
        for r in rows:
            ids = observer.prompt_ids(SYSTEM, r["user"])
            span = observer.content_indices(SYSTEM, r["user"], r["evidence"])
            group = partitions(len(ids), span)
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "partitions": group,
                "bins": bin_indices(group["Q"]),
            }
            if any(
                len(observer.candidate_ids(ids, r[key])) != 1
                for key in ("high_answer", "low_answer")
            ):
                raise ValueError("Single-token score required")
        for r in rows:
            target = layouts[r["case_id"]]
            donor = layouts[lookup[r["group_id"], r["variant"], r["form"], -r["level"]]]
            check_donor_layout(
                target["ids"], target["span"], donor["ids"], donor["span"]
            )
        write_json(out / "token_layouts.json", layouts)

        def budget(n=1):
            """Bound actual forwards and completed recipient cases."""
            if (
                observer.calls + n > 650
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1F finite packet budget")

        def record(r, logits):
            """Preserve candidate arithmetic for independent audit."""
            ids = layouts[r["case_id"]]["ids"]
            hi, lo = [
                observer.candidate_ids(ids, r[key])[0]
                for key in ("high_answer", "low_answer")
            ]
            return {
                k: r[k] for k in ("case_id", "group_id", "variant", "form", "level")
            } | {
                "margin": float(logits[hi]) - float(logits[lo]),
                "high_logit": float(logits[hi]),
                "low_logit": float(logits[lo]),
            }

        for r in rows:
            budget()
            case = r["case_id"]
            ids = layouts[case]["ids"]
            states[case], logits = observer.capture(
                ids, positions=list(range(len(ids)))
            )
            np.savez_compressed(out / "activations" / (case + ".npz"), **states[case])
            base.append(record(r, logits))
        flush("NATIVE_COMPLETE")
        for i, r in enumerate(rows):
            budget(11)
            case = r["case_id"]
            donor = lookup[r["group_id"], r["variant"], r["form"], -r["level"]]
            layout = layouts[case]
            patch_sets = [("block_04", "E", layout["partitions"]["E"])]
            patch_sets += [
                ("block_16", key, idx) for key, idx in layout["partitions"].items()
            ]
            patch_sets += [
                (
                    "block_16",
                    "EQD",
                    sorted(j for part in layout["partitions"].values() for j in part),
                )
            ]
            patch_sets += [
                ("block_16", f"bin{j + 1}", part)
                for j, part in enumerate(layout["bins"])
            ]
            if i == 0:
                budget(2)
                _, same = observer.capture(
                    layout["ids"],
                    patch=Patch(
                        "block_16",
                        layout["partitions"]["Q"],
                        states[case]["block_16"][layout["partitions"]["Q"]],
                    ),
                )
                _, restored = observer.capture(layout["ids"])
                error = max(
                    abs(record(r, same)["margin"] - base[0]["margin"]),
                    abs(record(r, restored)["margin"] - base[0]["margin"]),
                )
                checks.append({"self_restore_error": error})
                if error > 1e-5:
                    raise ValueError("Patch cleanup failed")
            for site, name, indices in patch_sets:
                budget()
                _, logits = observer.capture(
                    layout["ids"],
                    patch=Patch(site, indices, states[donor][site][indices]),
                )
                effects.append(
                    record(r, logits)
                    | {"site": site, "patch": name, "donor_case": donor}
                )
            if (i + 1) % 12 == 0:
                flush("PATCH_RUNNING")
                print(f"P1F {i + 1}/48 calls={observer.calls}", flush=True)
        tables = summarize(
            pd.DataFrame(base), pd.DataFrame(effects), layouts, states, shapes, models
        )
        err = float(
            tables["transport_rows"].query("patch == 'EQD'").restore_error.abs().max()
        )
        if err > 1e-5:
            raise ValueError("Full donor not restored")
        for name, frame in tables.items():
            frame.to_csv(out / (name + ".csv"), index=False)
        result = flush("COMPLETE", full_donor_error=err)
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
    """Run fresh finite material locally, without training or external text."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-dir", type=Path, required=True)
    p.add_argument("--shape-dir", type=Path, required=True)
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

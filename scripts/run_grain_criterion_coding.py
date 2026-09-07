#!/usr/bin/env python3
"""P1K: common entity state crossed with criterion polarity and answer coding."""

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


def material(parent):
    """Cross all P1J questions with antonyms and reversed answer convention."""
    clauses = {
        "event": {
            1: "did this person initiate the conversation rather than wait?",
            -1: "did this person wait rather than initiate the conversation?",
        },
        "trait": {
            1: "does this person's behavior seem outgoing rather than reserved?",
            -1: "does this person's behavior seem reserved rather than outgoing?",
        },
        "venue": {
            1: "was the room open during the event?",
            -1: "was the room closed during the event?",
        },
    }
    rows = []
    for r in parent:
        prefix = r["user"][: r["entity_span"][1]]
        for polarity in (1, -1):
            for code in (1, -1):
                convention = (
                    "Return A for Yes and B for No."
                    if code == 1
                    else "Return B for Yes and A for No."
                )
                user = (
                    prefix + ", " + clauses[r["criterion"]][polarity] + " " + convention
                )
                if polarity == code == 1 and user != r["user"]:
                    raise ValueError("Positive original prompt changed")
                rows.append(
                    {
                        **r,
                        "parent_case": r["case_id"],
                        "case_id": f"{r['case_id']}__p{polarity}__c{code}",
                        "polarity": polarity,
                        "code": code,
                        "user": user,
                        "expected_sign": polarity
                        * code
                        * (1 if r["criterion"] == "venue" else r["role_sign"]),
                    }
                )
    if len(rows) != 192 or len({r["case_id"] for r in rows}) != 192:
        raise ValueError("Complete192-case factorial required")
    return rows


def contrast_coefficients(values):
    """Invert the balanced 2x2 sign design exactly without model fitting."""
    if set(values) != {(p, c) for p in (-1, 1) for c in (-1, 1)}:
        raise ValueError("Missing polarity/code cell")
    weights = {
        "constant": lambda p, c: 1,
        "polarity": lambda p, c: p,
        "code": lambda p, c: c,
        "semantic": lambda p, c: p * c,
    }
    coefficients = {
        k: float(np.mean([v * f(p, c) for (p, c), v in values.items()]))
        for k, f in weights.items()
    }
    error = max(
        abs(v - sum(coefficients[k] * f(p, c) for k, f in weights.items()))
        for (p, c), v in values.items()
    )
    return coefficients, error


def summarize(base, effects):
    """Keep output preference, role response, exact parity and irrelevant effects."""
    lookup = base.set_index("case_id").margin
    rows = effects.copy()
    rows["shift"] = rows.margin - rows.case_id.map(lookup)
    rows["gap"] = rows.donor_case.map(lookup) - rows.case_id.map(lookup)
    rows["restore_error"] = rows["shift"] - rows.gap
    native, transport, contrasts, coefficients = [], [], [], []
    for (criterion, p, c), sub in base.groupby(["criterion", "polarity", "code"]):
        native.append(
            {
                "criterion": criterion,
                "polarity": p,
                "code": c,
                "cases": len(sub),
                "expected_preference": float(
                    np.mean(sub.margin * sub.expected_sign > 0)
                ),
            }
        )
    keys = ["group_id", "entity", "criterion", "polarity", "code"]
    for key, sub in base.groupby(keys):
        r = sub.set_index("role_sign").margin
        contrasts.append(
            dict(zip(keys, key))
            | {"channel": "native", "contrast": float(r[1] - r[-1])}
        )
    for key, sub in rows.groupby(keys + ["patch"]):
        contrasts.append(
            dict(zip(keys + ["channel"], key))
            | {"contrast": float(np.mean(-sub.role_sign * sub["shift"]))}
        )
    for (criterion, p, c, patch), sub in rows.groupby(
        ["criterion", "polarity", "code", "patch"]
    ):
        energy = float(np.sum(sub.gap**2))
        transport.append(
            {
                "criterion": criterion,
                "polarity": p,
                "code": c,
                "patch": patch,
                "cases": len(sub),
                "relative_error_energy": float(np.sum(sub.restore_error**2) / energy)
                if energy > 1e-12
                else None,
                "shift_rms": float(np.sqrt(np.mean(sub["shift"] ** 2))),
                "gap_rms": float(np.sqrt(np.mean(sub.gap**2))),
                "intended_fraction": float(
                    np.mean(-sub.role_sign * p * c * sub["shift"] > 0)
                )
                if criterion != "venue"
                else None,
            }
        )
    contrasts = pd.DataFrame(contrasts)
    for key, sub in contrasts.groupby(["group_id", "entity", "criterion", "channel"]):
        values = {(r.polarity, r.code): r.contrast for r in sub.itertuples()}
        coeff, err = contrast_coefficients(values)
        energy = sum(v * v for v in coeff.values())
        coefficients.append(
            dict(zip(["group_id", "entity", "criterion", "channel"], key))
            | coeff
            | {
                "reconstruction_error": err,
                "semantic_energy_share": coeff["semantic"] ** 2 / energy
                if energy > 1e-12
                else None,
            }
        )
    coefficients = pd.DataFrame(coefficients)
    summaries = []
    for key, sub in coefficients.groupby(["criterion", "channel"]):
        energy = float(
            np.sum(sub[["constant", "polarity", "code", "semantic"]].to_numpy() ** 2)
        )
        summaries.append(
            dict(zip(["criterion", "channel"], key))
            | {
                "pairs": len(sub),
                **{
                    k + "_mean": float(sub[k].mean())
                    for k in ("constant", "polarity", "code", "semantic")
                },
                **{
                    k + "_energy_share": float(np.sum(sub[k] ** 2) / energy)
                    if energy > 1e-12
                    else None
                    for k in ("constant", "polarity", "code", "semantic")
                },
                "positive_semantic_pairs": int(np.sum(sub.semantic > 0)),
            }
        )
    return {
        "response_rows": rows,
        "native_summary": pd.DataFrame(native),
        "transport_summary": pd.DataFrame(transport),
        "role_contrasts": contrasts,
        "coefficients": coefficients,
        "parity_summary": pd.DataFrame(summaries),
    }


def run(args):
    """Execute one immutable finite factorial, preserving all failed/native cases."""
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
            raise ValueError("Parent changed")
    data = json.loads((args.parent_dir / "materials.json").read_text())
    rows = material(data["rows"])
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "activations").mkdir()
    sources = [
        Path(__file__),
        ROOT / "docs/GRAIN_CRITERION_CODING_P1K_PLAN.md",
        ROOT / "tests/test_grain_criterion_coding.py",
    ] + [ROOT / p for p in parent["code"]]
    sources = list(dict.fromkeys(sources))
    manifest = {
        "parent_identity": parent["identity"],
        "material_hash": material_hash(rows),
        "private_data_read": False,
        "max_calls": 650,
        "max_seconds": 600,
        "pad_to_length": 256,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": data["system"], "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    base, effects, controls = [], [], []
    observer, start = None, time.monotonic()

    def flush(status, **extra):
        """Checkpoint raw rows and resource accounting without changing the analysis grid."""
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
        key_fields = ("group_id", "level", "entity", "criterion", "polarity", "code")
        lookup = {tuple(r[k] for k in key_fields): r["case_id"] for r in rows}
        for r in rows:
            ids = observer.prompt_ids(data["system"], r["user"])
            span = observer.content_indices(data["system"], r["user"], r["evidence"])
            name = observer.content_intervals(
                data["system"], r["user"], [r["entity_span"]]
            )
            if len(name) != 1 or any(
                len(observer.candidate_ids(ids, a)) != 1 for a in ("A", "B")
            ):
                raise ValueError("Name/code not one token")
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "N": name,
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
                    r["polarity"],
                    r["code"],
                ]
            ]
            check_donor_layout(l["ids"], l["span"], d["ids"], d["span"])
        write_json(out / "token_layouts.json", layouts)

        def budget():
            """Respect the frozen finite call/time/memory budget."""
            if (
                observer.calls >= 650
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1K resource cap")

        def record(r, logits):
            """Keep candidate logits and semantic/code signs separate."""
            a, b = [
                observer.candidate_ids(layouts[r["case_id"]]["ids"], v)[0]
                for v in ("A", "B")
            ]
            return {
                k: r[k]
                for k in (
                    *key_fields,
                    "case_id",
                    "parent_case",
                    "role_sign",
                    "expected_sign",
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
        prefix_error = 0.0
        prefix_checks = []
        for r in rows:
            case = r["case_id"]
            source = lookup[r["group_id"], r["level"], r["entity"], "event", 1, 1]
            n = layouts[case]["N"]
            if (
                n != layouts[source]["N"]
                or layouts[case]["ids"][: n[-1] + 1]
                != layouts[source]["ids"][: n[-1] + 1]
            ):
                raise ValueError("Common prefix changed")
            err = float(abs(states[case][n] - states[source][n]).max())
            prefix_error = max(prefix_error, err)
            prefix_checks.append(
                {"case_id": case, "source_case": source, "state_error": err}
            )
        pd.DataFrame(prefix_checks).to_csv(out / "prefix_checks.csv", index=False)
        old = pd.read_csv(args.parent_dir / "baselines.csv").set_index("case_id").margin
        canonical = pd.DataFrame(base).query("polarity == 1 and code == 1")
        replay = float(abs(canonical.margin - canonical.parent_case.map(old)).max())
        if max(replay, prefix_error) > 1e-5:
            raise ValueError("P1J replay or common prefix failed")
        flush(
            "NATIVE_COMPLETE", parent_replay_max=replay, prefix_state_max=prefix_error
        )
        for i, r in enumerate(rows):
            case, l = r["case_id"], layouts[r["case_id"]]
            donor = lookup[
                r["group_id"],
                -r["level"],
                r["entity"],
                r["criterion"],
                r["polarity"],
                r["code"],
            ]
            common = lookup[r["group_id"], -r["level"], r["entity"], "event", 1, 1]
            if i in (0, 12):
                budget()
                _, same = observer.capture(
                    l["ids"], patch=Patch("block_16", l["N"], states[case][l["N"]])
                )
                budget()
                _, restore = observer.capture(l["ids"])
                err = max(
                    abs(record(r, z)["margin"] - base[i]["margin"])
                    for z in (same, restore)
                )
                controls.append({"case_id": case, "self_restore_max": err})
                if err > 1e-5:
                    raise ValueError("Self/restore mismatch")
            for patch, indices in {
                "N": l["N"],
                "EQD": sorted(j for part in l["partitions"].values() for j in part),
            }.items():
                budget()
                source = common if patch == "N" else donor
                _, logits = observer.capture(
                    l["ids"], patch=Patch("block_16", indices, states[source][indices])
                )
                effects.append(
                    record(r, logits)
                    | {"patch": patch, "donor_case": donor, "state_source_case": source}
                )
            if (i + 1) % 48 == 0:
                flush(
                    "PATCH_RUNNING",
                    parent_replay_max=replay,
                    prefix_state_max=prefix_error,
                )
                print(f"P1K {i + 1}/192 calls={observer.calls}", flush=True)
        tables = summarize(pd.DataFrame(base), pd.DataFrame(effects))
        error = float(
            tables["response_rows"].query("patch == 'EQD'").restore_error.abs().max()
        )
        if observer.calls != 580 or len(effects) != 384 or error > 1e-5:
            raise ValueError("Complete packet/restoration failed")
        for name, table in tables.items():
            table.to_csv(out / (name + ".csv"), index=False)
        result = flush(
            "COMPLETE",
            parent_replay_max=replay,
            prefix_state_max=prefix_error,
            full_restore_max=error,
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
    """Run a single offline no-training factorial packet."""
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

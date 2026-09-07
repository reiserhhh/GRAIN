#!/usr/bin/env python3
"""P1N: cross behavioral and room evidence, testing target-dependent vector composition."""

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
    """Independently cross room truth with all original-order, explicit-target records."""
    rows = []
    for r in parent:
        if r["record_order"] != "original" or r["criterion"] == "venue":
            continue
        if r["evidence"].count("The room was open.") != 1:
            raise ValueError("Expected one original room fact")
        for room in (1, -1):
            evidence = (
                r["evidence"].replace("The room was open.", "The room was closed.")
                if room == -1
                else r["evidence"]
            )
            delta = len(evidence) - len(r["evidence"])
            rows.append(
                {
                    **r,
                    "case_id": r["case_id"] + f"__room{room}",
                    "parent_case": r["case_id"],
                    "room_sign": room,
                    "evidence": evidence,
                    "user": evidence + r["user"][len(r["evidence"]) :],
                    "anchor_span": [v + delta for v in r["anchor_span"]],
                    "expected_sign": room
                    if r["target_kind"] == "object"
                    else r["role_sign"],
                }
            )
    if len(rows) != 80 or len({r["user"] for r in rows}) != 80:
        raise ValueError("Complete80-case material required")
    return rows


def decompose(base, b, r, both, additive):
    """Separate finite representation departure and readout interaction on this path."""
    total = both - b - r + base
    state = both - additive
    readout = additive - b - r + base
    return {
        "total": total,
        "state_departure": state,
        "readout_interaction": readout,
        "identity_error": total - state - readout,
    }


def summarize(base, effects):
    """Retain relevant and irrelevant shifts, native truth performance, and signed interactions."""
    index = base.set_index("case_id").margin
    rows = effects.copy()
    rows["shift"] = rows.margin - rows.case_id.map(index)
    rows["gap"] = rows.native_donor_case.map(index) - rows.case_id.map(index)
    rows["restore_error"] = rows["shift"] - rows.gap
    summaries, contrasts, interactions = [], [], []
    for key, sub in rows.groupby(["criterion", "patch", "intervention"]):
        criterion, patch, intervention = key
        relevant = (
            intervention == "B"
            and criterion != "venue_object"
            or intervention == "R"
            and criterion == "venue_object"
            or intervention == "BR"
        )
        direction = sub.role_sign if criterion != "venue_object" else sub.room_sign
        summaries.append(
            {
                "criterion": criterion,
                "patch": patch,
                "intervention": intervention,
                "cases": len(sub),
                "shift_rms": float(np.sqrt(np.mean(sub["shift"] ** 2))),
                "mean_shift": float(sub["shift"].mean()),
                "relevant": relevant,
                "expected_direction_fraction": float(
                    np.mean(-direction * sub["shift"] > 0)
                )
                if relevant
                else None,
            }
        )
    for axis, other in (("B", "room_sign"), ("R", "level")):
        for key, sub in base.groupby(["group_id", "entity", "criterion", other]):
            values = sub.set_index("role_sign" if axis == "B" else "room_sign").margin
            contrasts.append(
                dict(zip(["group_id", "entity", "criterion", "fixed_value"], key))
                | {"axis": axis, "contrast": float(values[1] - values[-1])}
            )
    for case, sub in rows.groupby("case_id"):
        lookup = sub.set_index(["patch", "intervention"]).margin
        score = decompose(
            index[case],
            lookup["anchor", "B"],
            lookup["anchor", "R"],
            lookup["anchor", "BR"],
            lookup["anchor", "ADD"],
        )
        r = base.set_index("case_id").loc[case]
        interactions.append(
            {
                "case_id": case,
                "group_id": r.group_id,
                "entity": r.entity,
                "criterion": r.criterion,
                "level": r.level,
                "room_sign": r.room_sign,
                **score,
                "native_total": float(
                    lookup["EQD", "BR"]
                    - lookup["EQD", "B"]
                    - lookup["EQD", "R"]
                    + index[case]
                ),
            }
        )
    interactions = pd.DataFrame(interactions)
    energy = []
    for criterion, sub in interactions.groupby("criterion"):
        energy.append(
            {
                "criterion": criterion,
                "cases": len(sub),
                "total_sse": float(np.sum(sub.total**2)),
                "state_sse": float(np.sum(sub.state_departure**2)),
                "readout_sse": float(np.sum(sub.readout_interaction**2)),
                "cross_term": float(
                    2 * np.sum(sub.state_departure * sub.readout_interaction)
                ),
                "native_total_sse": float(np.sum(sub.native_total**2)),
            }
        )
    native = (
        base.groupby(["criterion", "room_sign"])
        .apply(
            lambda s: pd.Series(
                {
                    "cases": len(s),
                    "expected_preference": float(
                        np.mean(s.margin * s.expected_sign > 0)
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return {
        "response_rows": rows,
        "effect_summary": pd.DataFrame(summaries),
        "native_contrasts": pd.DataFrame(contrasts),
        "native_summary": native,
        "interactions": interactions,
        "interaction_energy": pd.DataFrame(energy),
    }


def run(args):
    """Execute the bounded80-world panel with natural and additive-anchor interventions."""
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    if (
        json.loads((args.parent_dir / "run_status.json").read_text())["status"]
        != "COMPLETE"
    ):
        raise ValueError("Complete P1M required")
    for name, digest in json.loads(
        (args.parent_dir / "output_hashes.json").read_text()
    ).items():
        if file_hash(args.parent_dir / name) != digest:
            raise ValueError("Parent artifact changed")
    data = json.loads((args.parent_dir / "materials.json").read_text())
    rows = material(data["rows"])
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "activations").mkdir()
    sources = list(
        dict.fromkeys(
            [
                Path(__file__),
                ROOT / "docs/GRAIN_ATTRIBUTE_ROUTING_P1N_PLAN.md",
                ROOT / "tests/test_grain_attribute_routing.py",
            ]
            + [ROOT / p for p in parent["code"]]
        )
    )
    manifest = {
        "parent_identity": parent["identity"],
        "material_hash": material_hash(rows),
        "max_calls": 750,
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
    base, effects, controls, geometry = [], [], [], []
    observer, start = None, time.monotonic()

    def flush(status, **extra):
        """Preserve every score and resource limit; never promote a partial factorial."""
        pd.DataFrame(base).to_csv(out / "baselines.csv", index=False)
        pd.DataFrame(effects).to_csv(out / "effects.csv", index=False)
        pd.DataFrame(geometry).to_csv(out / "anchor_geometry.csv", index=False)
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
        keys = ("group_id", "level", "entity", "criterion", "room_sign")
        lookup = {tuple(r[k] for k in keys): r["case_id"] for r in rows}
        layouts, states, donors = {}, {}, {}
        for r in rows:
            ids = observer.prompt_ids(data["system"], r["user"])
            span = observer.content_indices(data["system"], r["user"], r["evidence"])
            anchor = observer.content_intervals(
                data["system"], r["user"], [r["anchor_span"]]
            )
            if len(anchor) != 1 or any(
                len(observer.candidate_ids(ids, a)) != 1 for a in ("A", "B")
            ):
                raise ValueError("Expected one-token anchor/answers")
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "anchor": anchor,
                "partitions": partitions(len(ids), span),
            }
            donors[r["case_id"]] = {}
            for label, b, t in (
                ("B", -r["level"], r["room_sign"]),
                ("R", r["level"], -r["room_sign"]),
                ("BR", -r["level"], -r["room_sign"]),
            ):
                donors[r["case_id"]][label] = {
                    "full": lookup[r["group_id"], b, r["entity"], r["criterion"], t],
                    "anchor": lookup[
                        r["group_id"],
                        b,
                        r["entity"],
                        "event" if r["target_kind"] == "person" else "venue_object",
                        t,
                    ],
                }
        for case, variants in donors.items():
            l = layouts[case]
            for record in variants.values():
                d = layouts[record["full"]]
                check_donor_layout(l["ids"], l["span"], d["ids"], d["span"])
                if layouts[record["anchor"]]["anchor"] != l["anchor"]:
                    raise ValueError("Anchor alignment changed")
        write_json(out / "token_layouts.json", layouts)
        write_json(out / "donors.json", donors)

        def budget():
            """Enforce one frozen resource budget before each model call."""
            if (
                observer.calls >= 750
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1N resource cap")

        def record(r, logits):
            """Record candidate logits separately from authored truth and role signs."""
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
                    "role_sign",
                    "expected_sign",
                    "target_kind",
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
                r["room_sign"],
            ]
            n = layouts[case]["anchor"]
            if layouts[case]["ids"][: n[-1] + 1] != layouts[source]["ids"][: n[-1] + 1]:
                raise ValueError("Common prefix differs")
            checks.append(
                {
                    "case_id": case,
                    "source_case": source,
                    "state_error": float(
                        abs(states[case][n] - states[source][n]).max()
                    ),
                }
            )
        prefix_error = max(c["state_error"] for c in checks)
        pd.DataFrame(checks).to_csv(out / "prefix_checks.csv", index=False)
        canonical = pd.DataFrame(base).query("room_sign == 1")
        original = (
            pd.read_csv(args.parent_dir / "baselines.csv").set_index("case_id").margin
        )
        replay = float(
            abs(canonical.margin - canonical.parent_case.map(original)).max()
        )
        if len(canonical) != 40 or max(prefix_error, replay) > 1e-5:
            raise ValueError("Prefix/native replay failed")
        flush(
            "NATIVE_COMPLETE", prefix_state_max=prefix_error, parent_replay_max=replay
        )
        checked = set()
        for i, r in enumerate(rows):
            case, l = r["case_id"], layouts[r["case_id"]]
            n = l["anchor"]
            if r["target_kind"] not in checked:
                checked.add(r["target_kind"])
                budget()
                _, same = observer.capture(
                    l["ids"], patch=Patch("block_16", n, states[case][n])
                )
                budget()
                _, restore = observer.capture(l["ids"])
                error = max(
                    abs(record(r, z)["margin"] - base[i]["margin"])
                    for z in (same, restore)
                )
                controls.append({"case_id": case, "self_restore_max": error})
                if error > 1e-5:
                    raise ValueError("Self/restore mismatch")
            variants = donors[case]
            for label, sources in variants.items():
                for patch, positions in {
                    "anchor": n,
                    "EQD": sorted(j for part in l["partitions"].values() for j in part),
                }.items():
                    budget()
                    source = sources["anchor"] if patch == "anchor" else sources["full"]
                    _, logits = observer.capture(
                        l["ids"],
                        patch=Patch("block_16", positions, states[source][positions]),
                    )
                    effects.append(
                        record(r, logits)
                        | {
                            "patch": patch,
                            "intervention": label,
                            "native_donor_case": sources["full"],
                            "state_source_case": source,
                        }
                    )
            n0, nb, nr, nbr = (
                states[case][n],
                *(states[variants[k]["anchor"]][n] for k in ("B", "R", "BR")),
            )
            add = nb + nr - n0
            if not np.isfinite(add).all():
                raise ValueError("Nonfinite additive state")
            geometry.append(
                {
                    "case_id": case,
                    "base_norm": float(np.linalg.norm(n0)),
                    "add_norm": float(np.linalg.norm(add)),
                    "B_displacement": float(np.linalg.norm(nb - n0)),
                    "R_displacement": float(np.linalg.norm(nr - n0)),
                    "BR_displacement": float(np.linalg.norm(nbr - n0)),
                    "mixed_state_norm": float(np.linalg.norm(nbr - add)),
                }
            )
            budget()
            _, logits = observer.capture(l["ids"], patch=Patch("block_16", n, add))
            effects.append(
                record(r, logits)
                | {
                    "patch": "anchor",
                    "intervention": "ADD",
                    "native_donor_case": "",
                    "state_source_case": "synthetic_B_plus_R_minus_base",
                }
            )
            if (i + 1) % 20 == 0:
                flush(
                    "PATCH_RUNNING",
                    prefix_state_max=prefix_error,
                    parent_replay_max=replay,
                )
                print(f"P1N {i + 1}/80 calls={observer.calls}", flush=True)
        tables = summarize(pd.DataFrame(base), pd.DataFrame(effects))
        restore = float(
            tables["response_rows"].query("patch == 'EQD'").restore_error.abs().max()
        )
        if observer.calls != 644 or len(effects) != 560 or restore > 1e-5:
            raise ValueError("Incomplete packet or full restore failed")
        for name, table in tables.items():
            table.to_csv(out / (name + ".csv"), index=False)
        result = flush(
            "COMPLETE",
            prefix_state_max=prefix_error,
            parent_replay_max=replay,
            full_restore_max=restore,
            identity_max=float(tables["interactions"].identity_error.abs().max()),
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
    """Run one complete offline attribute-selection/composition packet."""
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

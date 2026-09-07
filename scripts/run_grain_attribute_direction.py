#!/usr/bin/env python3
"""P1P: frozen rank-one attribute interventions on new synthetic evidence."""

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

SCENES = (
    ("p_museum", "museum guide briefing", 0, 1, ("Alex", "Blair")),
    ("p_books", "neighborhood book exchange", 1, -1, ("Blair", "Alex")),
    ("p_cooking", "weekend cooking workshop", 1, 1, ("Blair", "Alex")),
    ("p_photography", "amateur photography meetup", 0, -1, ("Alex", "Blair")),
)
ACTIONS = (
    (
        "walked up to a newcomer, introduced themselves, and started a conversation.",
        "waited quietly near a newcomer until the newcomer started a conversation.",
    ),
    (
        "approached an unfamiliar attendee and opened a conversation by asking a question.",
        "listened quietly near an unfamiliar attendee and waited for them to open a conversation.",
    ),
)


def material(parent):
    """Build all held records, preserving question suffixes rather than tuning them."""
    templates = {}
    for r in parent:
        templates.setdefault((r["entity"], r["criterion"]), r)
    rows = []
    for group, scene, family, assignment, order in SCENES:
        for level in (-1, 1):
            role = {"Alex": assignment * level, "Blair": -assignment * level}
            for room in (-1, 1):
                evidence = f"Record of a {scene}. The room was {'open' if room == 1 else 'closed'}.\n"
                evidence += "\n".join(
                    name + " " + ACTIONS[family][0 if role[name] == 1 else 1]
                    for name in order
                )
                for (entity, criterion), t in sorted(templates.items()):
                    suffix = t["user"][len(t["evidence"]) :]
                    change = len(evidence) - len(t["evidence"])
                    sign = role.get(entity, level)
                    rows.append(
                        {
                            "case_id": f"{group}__{level}__{room}__{entity}__{criterion}",
                            "group_id": group,
                            "level": level,
                            "room_sign": room,
                            "entity": entity,
                            "criterion": criterion,
                            "role_sign": sign,
                            "expected_sign": room if entity == "room" else sign,
                            "target_kind": "object" if entity == "room" else "person",
                            "action_family": family,
                            "evidence": evidence,
                            "user": evidence + suffix,
                            "anchor_span": [v + change for v in t["anchor_span"]],
                            "anchor_text": entity,
                        }
                    )
    if len(rows) != 80 or len({r["user"] for r in rows}) != 80:
        raise ValueError("Held material must contain80 distinct prompts")
    return rows


def make_directions(coefficients, order):
    """Estimate one mean effect per target using training vectors only."""
    result = {}
    rng = np.random.default_rng(20260907)
    for entity in ("Alex", "Blair", "room"):
        vectors = [
            coefficients[i, 2 if entity == "room" else 1]
            for i, r in enumerate(order)
            if r["entity"] == entity
        ]
        if len(vectors) != 4:
            raise ValueError("Four training settings per target required")
        v = np.mean(vectors, axis=0)
        if np.linalg.norm(v) < 1e-10:
            raise ValueError("Degenerate training direction")
        d = v / np.linalg.norm(v)
        q = rng.normal(size=len(d))
        q -= np.dot(q, d) * d
        q /= np.linalg.norm(q)
        result[entity] = {"direction": d, "random": q}
    return result


def components(n0, donor, d, q):
    """Split a finite displacement, with an equal-norm orthogonal control."""
    x, y = np.asarray(n0, dtype=float), np.asarray(donor, dtype=float)
    delta = y - x
    amplitude = float(np.dot(d, delta))
    projected = amplitude * d
    return {
        "natural": y,
        "projected": x + projected,
        "complement": y - projected,
        "matched_random": x + amplitude * q,
    }


def summarize(base, effects):
    """Expose all relevant/irrelevant effects; no rank or direction selection."""
    zero = base.set_index("case_id").margin
    rows = effects.copy()
    rows["shift"] = rows.margin - rows.case_id.map(zero)
    summaries = []
    for scope, keys in (
        ("setting", ["group_id", "criterion", "axis"]),
        ("pooled", ["criterion", "axis"]),
    ):
        for key, sub in rows.groupby(keys):
            info = dict(zip(keys, key))
            pivot = sub.pivot(index="case_id", columns="patch", values="shift")
            rms = float(np.sqrt(np.mean(pivot.natural**2)))
            relevant = info["axis"] == (
                "R" if info["criterion"] == "venue_object" else "B"
            )
            for patch in pivot:
                values = pivot[patch]
                meta = sub[sub.patch == patch].set_index("case_id").loc[pivot.index]
                sign = (
                    meta.room_sign
                    if info["criterion"] == "venue_object"
                    else meta.role_sign
                )
                err = float(np.sqrt(np.mean((values - pivot.natural) ** 2)))
                size = float(np.sqrt(np.mean(values**2)))
                summaries.append(
                    {
                        **info,
                        "scope": scope,
                        "patch": patch,
                        "cases": len(values),
                        "relevant": relevant,
                        "natural_rms": rms,
                        "shift_rms": size,
                        "error_rms": err,
                        "relative_rms": size / rms if rms > 1e-10 else np.nan,
                        "relative_error": err / rms if rms > 1e-10 else np.nan,
                        "intended_fraction": float(np.mean(-sign * values > 0))
                        if relevant
                        else np.nan,
                    }
                )
    return rows, pd.DataFrame(summaries)


def run(args):
    """Execute the pre-frozen minimal complete panel, preserving failed conditions."""
    for folder in (args.parent_dir, args.geometry_dir):
        if json.loads((folder / "run_status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Complete parent required")
        for name, digest in json.loads(
            (folder / "output_hashes.json").read_text()
        ).items():
            if file_hash(folder / name) != digest:
                raise ValueError("Changed parent")
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    geometry_manifest = json.loads((args.geometry_dir / "manifest.json").read_text())
    if geometry_manifest["parent_identity"] != parent["identity"]:
        raise ValueError("Wrong geometry parent")
    data = json.loads((args.parent_dir / "materials.json").read_text())
    rows = material(data["rows"])
    directions = make_directions(
        np.load(args.geometry_dir / "coefficients.npz")["coefficients"],
        json.loads((args.geometry_dir / "coefficient_index.json").read_text()),
    )
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("activations", "patch_states"):
        (out / folder).mkdir()
    sources = list(
        dict.fromkeys(
            [
                Path(__file__),
                ROOT / "docs/GRAIN_ATTRIBUTE_DIRECTION_P1P_PLAN.md",
                ROOT / "tests/test_grain_attribute_direction.py",
            ]
            + [ROOT / p for p in parent["code"]]
        )
    )
    manifest = {
        "parent_identity": parent["identity"],
        "geometry_manifest_sha256": file_hash(args.geometry_dir / "manifest.json"),
        "material_hash": material_hash(rows),
        "max_calls": 800,
        "max_seconds": 600,
        "training_settings": 4,
        "held_settings": 4,
        "private_data_read": False,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": data["system"], "rows": rows})
    np.savez_compressed(
        out / "directions.npz",
        **{f"{e}_{k}": v for e, pair in directions.items() for k, v in pair.items()},
    )
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    base, effects, checks, geometry = [], [], [], []
    observer, start = None, time.monotonic()

    def flush(status, **extra):
        """Checkpoint progress without turning an incomplete panel into a result."""
        for name, values in (
            ("baselines", base),
            ("effects", effects),
            ("geometry", geometry),
        ):
            pd.DataFrame(values).to_csv(out / (name + ".csv"), index=False)
        write_json(out / "instrument_checks.json", checks)
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
        keys = ("group_id", "level", "entity", "criterion", "room_sign")
        lookup = {tuple(r[k] for k in keys): r["case_id"] for r in rows}
        layouts, donors, states = {}, {}, {}
        for r in rows:
            ids = observer.prompt_ids(data["system"], r["user"])
            span = observer.content_indices(data["system"], r["user"], r["evidence"])
            anchor = observer.content_intervals(
                data["system"], r["user"], [r["anchor_span"]]
            )
            if len(anchor) != 1 or any(
                len(observer.candidate_ids(ids, v)) != 1 for v in ("A", "B")
            ):
                raise ValueError("Expected single-token target/answers")
            layouts[r["case_id"]] = {
                "ids": ids,
                "span": span,
                "anchor": anchor,
                "partitions": partitions(len(ids), span),
            }
            donors[r["case_id"]] = {}
            for axis, b, t in (
                ("B", -r["level"], r["room_sign"]),
                ("R", r["level"], -r["room_sign"]),
            ):
                donors[r["case_id"]][axis] = {
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
                    raise ValueError("Anchor positions differ")
        write_json(out / "token_layouts.json", layouts)
        write_json(out / "donors.json", donors)

        def budget():
            """Bound calls, time and allocated device memory before every forward."""
            if (
                observer.calls >= 800
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1P resource cap")

        def record(r, logits):
            """Preserve raw candidate logits and authored truth separately."""
            a, b = [
                observer.candidate_ids(layouts[r["case_id"]]["ids"], v)[0]
                for v in ("A", "B")
            ]
            return {
                k: r[k]
                for k in (*keys, "case_id", "role_sign", "expected_sign", "target_kind")
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
        canonical = {}
        for r in rows:
            source = lookup[
                r["group_id"],
                r["level"],
                r["entity"],
                "event" if r["target_kind"] == "person" else "venue_object",
                r["room_sign"],
            ]
            n = layouts[r["case_id"]]["anchor"]
            if (
                layouts[r["case_id"]]["ids"][: n[-1] + 1]
                != layouts[source]["ids"][: n[-1] + 1]
            ):
                raise ValueError("Different common prefix")
            prefix_error = max(
                prefix_error,
                float(abs(states[r["case_id"]][n] - states[source][n]).max()),
            )
            canonical[r["case_id"]] = source
        if prefix_error > 1e-5:
            raise ValueError("Shared-state identity failed")
        flush("NATIVE_COMPLETE", prefix_state_max=prefix_error)
        checked = set()
        for i, r in enumerate(rows):
            case, l = r["case_id"], layouts[r["case_id"]]
            n = l["anchor"]
            n0 = states[canonical[case]][n][0]
            d, q = (directions[r["entity"]][k] for k in ("direction", "random"))
            if r["entity"] not in checked:
                checked.add(r["entity"])
                for mode in ("identity", "restore"):
                    budget()
                    _, logits = observer.capture(
                        l["ids"],
                        patch=Patch("block_16", n, n0[None, :])
                        if mode == "identity"
                        else None,
                    )
                    error = abs(record(r, logits)["margin"] - base[i]["margin"])
                    checks.append({"case_id": case, "check": mode, "error": error})
                    if error > 1e-5:
                        raise ValueError("Identity/restore failed")
                for axis in ("B", "R"):
                    positions = sorted(
                        j for part in l["partitions"].values() for j in part
                    )
                    source = donors[case][axis]["full"]
                    budget()
                    _, logits = observer.capture(
                        l["ids"],
                        patch=Patch("block_16", positions, states[source][positions]),
                    )
                    ref = next(v["margin"] for v in base if v["case_id"] == source)
                    error = abs(record(r, logits)["margin"] - ref)
                    checks.append(
                        {"case_id": case, "check": "full_" + axis, "error": error}
                    )
                    if error > 1e-5:
                        raise ValueError("Full restore failed")
            for axis in ("B", "R"):
                source = donors[case][axis]["anchor"]
                value = states[source][n][0]
                patches = components(n0, value, d, q)
                saved = {k: v.astype(np.float32) for k, v in patches.items()}
                np.savez_compressed(
                    out / "patch_states" / (case + "__" + axis + ".npz"), **saved
                )
                delta = value.astype(float) - n0.astype(float)
                geometry.append(
                    {
                        "case_id": case,
                        "entity": r["entity"],
                        "axis": axis,
                        "delta_norm": float(np.linalg.norm(delta)),
                        "projected_norm": float(abs(np.dot(d, delta))),
                        "projection_energy_fraction": float(
                            np.dot(d, delta) ** 2 / np.dot(delta, delta)
                        )
                        if np.dot(delta, delta) > 1e-10
                        else np.nan,
                        "fp32_cast_error": max(
                            float(abs(saved[k].astype(float) - patches[k]).max())
                            for k in saved
                        ),
                    }
                )
                for patch, vector in saved.items():
                    budget()
                    _, logits = observer.capture(
                        l["ids"], patch=Patch("block_16", n, vector[None, :])
                    )
                    effects.append(
                        record(r, logits)
                        | {"axis": axis, "patch": patch, "source_case": source}
                    )
            if (i + 1) % 20 == 0:
                print(
                    json.dumps(flush("PATCHING", completed_settings=(i + 1) // 20)),
                    flush=True,
                )
        if len(base) != 80 or len(effects) != 640 or observer.calls != 732:
            raise ValueError("Incomplete declared panel")
        response, summary = summarize(pd.DataFrame(base), pd.DataFrame(effects))
        response.to_csv(out / "response_rows.csv", index=False)
        summary.to_csv(out / "effect_summary.csv", index=False)
        status = flush(
            "COMPLETE",
            prefix_state_max=prefix_error,
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
            traceback=traceback.format_exc(),
        )
        raise


def main():
    """Run only from explicit local immutable inputs and a fresh output directory."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-dir", "geometry-dir", "model-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()

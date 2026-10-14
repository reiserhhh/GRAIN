#!/usr/bin/env python3
"""HM10B2: fresh crossed target evidence and ownership response laws."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_native_joint as N
from grain_research.construction import math_target_orbits as O
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_native_joint import collect, frozen_files, write_json
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_target_orbits_v1.json"
PLAN = ROOT / "docs/GRAIN_TARGET_EVIDENCE_ORBIT_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/target_orbits_r1"
MAIN = [1, 2, 4, 8, 16]
BINDING = MAIN + [5, 10]
CODE = (
    "scripts/run_grain_math_target_orbits.py",
    "grain_research/construction/math_target_orbits.py",
    "scripts/run_grain_math_native_joint.py",
    "grain_research/construction/math_native_joint.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/vector_mechanisms/full_state_edit.py",
    "grain_research/vector_mechanisms/functional_profiles.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_narrow_functions.py",
)


def prepare(out):
    """Freeze new raw materials and source units before any native observations."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("preserve prior prepared packet")
    spec = read_json(SPEC)
    reference = ROOT / spec["reference"]
    old = read_json(reference / "manifest.json")
    ref = read_json(reference / "reference_manifest.json")
    if old["meps_reference_authors"] != 0 or old["human_outcome_fit"]:
        raise ValueError("private construction forbidden")
    if (
        digest(reference / "reference.npz") != ref["sha256"]
        or digest(MODEL / "model.safetensors") != old["model_sha256"]
    ):
        raise ValueError("frozen source observer changed")
    docs = O.material_bank(spec)
    previous = ROOT / spec["previous_packet"]
    hashes = read_json(previous / "03/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(previous / name) != expected:
            raise ValueError("prior packet changed")
    old_texts = {r["text"] for r in read_json(previous / "00/documents.json")}
    if any(r["text"] in old_texts for r in docs):
        raise ValueError("new raw material duplicates previous packet")
    by_text = {}
    for row in docs:
        by_text.setdefault(row["text"], len(by_text))
        row["text_id"] = by_text[row["text"]]
    if len(docs) != spec["document_records"] or len(by_text) != spec["unique_texts"]:
        raise ValueError("material coverage differs from frozen design")
    definitions = read_json(reference / "queries.json")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design = []
    for text, text_id in by_text.items():
        for query in spec["queries"]:
            _, _, encoded, mask = reference_view(
                tokenizer, [text], definitions[query]["question"]
            )
            if (
                np.asarray(encoded["input_ids"])[mask].tolist()
                != old["question_tokens"][query]
            ):
                raise ValueError("query token coordinates changed")
            design.append(
                {
                    "unit": len(design),
                    "text_id": text_id,
                    "query": query,
                    "input_ids": encoded["input_ids"],
                    "mask": mask.tolist(),
                }
            )
    if len(design) != spec["forward_calls"]:
        raise ValueError("forward count differs from frozen design")
    (out / "01/units").mkdir(parents=True)
    write_json(out / "00/documents.json", docs)
    write_json(out / "00/design.json", design)
    write_json(out / "00/queries.json", {q: definitions[q] for q in spec["queries"]})
    sources = {
        name: digest(ROOT / name)
        for name in (
            *CODE,
            str(SPEC.relative_to(ROOT)),
            str(PLAN.relative_to(ROOT)),
            str((previous / "03/artifact_hashes.json").relative_to(ROOT)),
            *(
                str((reference / name).relative_to(ROOT))
                for name in (
                    "queries.json",
                    "manifest.json",
                    "reference_manifest.json",
                    "reference.npz",
                )
            ),
        )
    }
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "packet": spec["packet"],
            "spec": spec,
            "model_sha256": old["model_sha256"],
            "alphabet": old["alphabet"],
            "question_tokens": {q: old["question_tokens"][q] for q in spec["queries"]},
            "inputs": {
                name: digest(out / name)
                for name in ("00/documents.json", "00/design.json", "00/queries.json")
            },
            "source_files": sources,
            "previous_hashes_checked": len(hashes),
            "frozen_before_new_calls": True,
            "new_unique_texts": len(by_text),
            "no_overlap_with_previous_materials": True,
            "private_meps_reads": 0,
            "api_calls": 0,
            "new_neural_training": False,
            "human_truth": False,
            "qualification_is_declared_not_automatically_parsed": True,
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "units": 0, "seconds": 0},
    )
    print(
        {
            "status": "PREPARED",
            "records": len(docs),
            "unique_texts": len(by_text),
            "forwards": len(design),
        },
        flush=True,
    )


def family_interval(coefficients, app_logits, app_actual, values, seed, draws):
    """Re-estimate source coefficients and independently resample cold families."""
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(draws):
        c = coefficients[rng.integers(0, len(coefficients), len(coefficients))].mean(
            axis=0
        )
        main, binding = [], []
        for logits, actual in zip(app_logits, app_actual, strict=True):
            main.append(
                (
                    N.score(O.anchored_completion(logits[0], c, MAIN), values)[1:]
                    - actual[1:]
                )
                ** 2
            )
            binding.append(
                (
                    N.score(O.anchored_completion(logits[0], c, BINDING), values)[1:]
                    - actual[1:]
                )
                ** 2
            )
        selected = rng.integers(0, len(app_logits), len(app_logits))
        gains.append(float((np.asarray(main) - np.asarray(binding))[selected].mean()))
    return np.quantile(gains, [0.025, 0.975]).tolist()


def evaluate(out):
    """Preserve all responses; compute binding, group components and cold actions."""
    manifest = frozen_files(out)
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("actual observations incomplete")
    spec = manifest["spec"]
    values = np.asarray(manifest["alphabet"]["values"])
    docs = read_json(out / "00/documents.json")
    units = {}
    for row in read_json(out / "00/design.json"):
        path = out / "01/units" / f"unit_{row['unit']:04d}.json"
        meta = read_json(path)
        if meta["array_sha256"] != digest(path.with_suffix(".npz")):
            raise ValueError("native unit changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            logits = arrays["logits"].astype(float)
        units[row["text_id"], row["query"]] = {
            "native": meta["native"]["conditional_score"],
            "affine": meta["affine"]["score"],
            "logits": logits,
            "candidate_mass": meta["native"]["candidate_mass"],
        }
    readouts, tables, controls = [], {}, []
    for doc in docs:
        for query in spec["queries"]:
            unit = units[doc["text_id"], query]
            readouts.append(
                {
                    "material_id": doc["material_id"],
                    "family": doc["family"],
                    "role": doc["role"],
                    "kind": doc["kind"],
                    "mask": doc["mask"],
                    "text_id": doc["text_id"],
                    "query": query,
                    "evidence_status": doc["evidence_status"][query],
                    "native": unit["native"],
                    "affine": unit["affine"],
                    "candidate_mass": unit["candidate_mass"],
                    "qualified_native": unit["native"]
                    if doc["evidence_status"][query].startswith("direct_")
                    else None,
                }
            )
            if doc["kind"] == "full":
                tables.setdefault((doc["family"], query), {})[doc["mask"]] = (doc, unit)
    for family in range(16):
        for query in spec["queries"]:
            group = [r for r in docs if r["family"] == family and r["kind"] != "full"]
            keyed = defaultdict(list)
            for row in group:
                keyed[
                    row["kind"],
                    row["anxiety_polarity"],
                    row["trust_polarity"],
                    row["anxiety_owner"],
                    row["trust_owner"],
                ].append(row)
            for pair in keyed.values():
                if len(pair) != 2 or pair[0]["text"] != pair[1]["text"]:
                    raise ValueError("single-target order null changed")
                controls.append(
                    {
                        "family": family,
                        "query": query,
                        "kind": pair[0]["kind"],
                        "left": pair[0]["material_id"],
                        "right": pair[1]["material_id"],
                        "native_delta": units[pair[1]["text_id"], query]["native"]
                        - units[pair[0]["text_id"], query]["native"],
                    }
                )
    coefficients, effects, bindings, edges = [], [], [], []
    natural_tables, scalar_tables = {}, {}
    max_reconstruction = max_energy = max_interaction = max_integral = 0.0
    for (family, query), cells in tables.items():
        if set(cells) != set(range(32)):
            raise ValueError("incomplete five-factor table")
        role = cells[0][0]["role"]
        logits = np.array([cells[m][1]["logits"] for m in range(32)])
        theta = N.center_logits(logits)
        native = np.array([cells[m][1]["native"] for m in range(32)])
        affine = np.array([cells[m][1]["affine"] for m in range(32)])
        parts = {
            "natural": O.decompose(theta),
            "native": O.decompose(native),
            "affine": O.decompose(affine),
        }
        natural_tables[family, query] = {
            "family": family,
            "role": role,
            "values": theta,
            "coefficients": parts["natural"]["coefficients"],
        }
        scalar_tables[family, query] = {
            "family": family,
            "role": role,
            "values": native,
            "coefficients": parts["native"]["coefficients"],
        }
        for part in parts.values():
            max_reconstruction = max(max_reconstruction, part["reconstruction_error"])
            max_energy = max(max_energy, part["energy_closure_error"])
        for term in range(32):
            coefficients.append(
                {
                    "family": family,
                    "role": role,
                    "query": query,
                    "term": term,
                    "natural_coefficient": parts["natural"]["coefficients"][
                        term
                    ].tolist(),
                    **{
                        f"{name}_energy": parts[name]["energies"][term]
                        for name in parts
                    },
                    "native_coefficient": float(parts["native"]["coefficients"][term]),
                    "affine_coefficient": float(parts["affine"]["coefficients"][term]),
                }
            )
        for axis, polarity_bit, owner_bit, term in (
            ("anxiety", 1, 4, 5),
            ("trust", 2, 8, 10),
        ):
            delta = {}
            for owner in (0, 1):
                selected = {
                    p: [
                        native[m]
                        for m in range(32)
                        if bool(m & polarity_bit) == bool(p)
                        and bool(m & owner_bit) == bool(owner)
                    ]
                    for p in (0, 1)
                }
                delta[owner] = float(np.mean(selected[1]) - np.mean(selected[0]))
            contrast = delta[0] - delta[1]
            max_interaction = max(
                max_interaction,
                abs(contrast + 4 * parts["native"]["coefficients"][term]),
            )
            bindings.append(
                {
                    "family": family,
                    "role": role,
                    "query": query,
                    "axis": axis,
                    "self_polarity_delta": delta[0],
                    "other_polarity_delta": delta[1],
                    "binding_contrast": contrast,
                    "affine_binding": -4 * float(parts["affine"]["coefficients"][term]),
                    "natural_binding_norm": float(
                        4 * np.linalg.norm(parts["natural"]["coefficients"][term])
                    ),
                }
            )
            for anchor in range(32):
                if anchor & (polarity_bit | owner_bit):
                    continue
                masks = [
                    anchor,
                    anchor | polarity_bit,
                    anchor | owner_bit,
                    anchor | polarity_bit | owner_bit,
                ]
                action = N.interaction(logits[masks], values)
                integral = N.response_integral(
                    action["additive_logits"], action["logit_interaction"], values
                )
                max_integral = max(
                    max_integral, abs(integral - action["representation_action"])
                )
                effects.append(
                    {
                        "family": family,
                        "role": role,
                        "query": query,
                        "axis": axis,
                        "anchor_mask": anchor,
                        "binding_contrast": -action["total"],
                        "link_binding": -action["link_only"],
                        "representation_binding": -action["representation_action"],
                        "natural_interaction_norm": float(
                            np.linalg.norm(action["logit_interaction"])
                        ),
                    }
                )
        for bit in (4, 8, 16):
            for mask in range(32):
                if mask & bit:
                    continue
                first, second = cells[mask], cells[mask ^ bit]
                if O.wordbag(first[0]["text"]) != O.wordbag(second[0]["text"]):
                    raise ValueError("orbit edge changed words")
                edges.append(
                    {
                        "family": family,
                        "query": query,
                        "generator": bit,
                        "mask": mask,
                        "native_delta": second[1]["native"] - first[1]["native"],
                        "affine_delta": second[1]["affine"] - first[1]["affine"],
                    }
                )
    predictions, completion, intervals, binding_summary, evidence_summary = (
        [],
        [],
        [],
        [],
        [],
    )
    for qi, query in enumerate(spec["queries"]):
        source = [r for (f, q), r in natural_tables.items() if q == query]
        c = O.source_coefficients(source)
        cs = O.source_coefficients(
            [r for (f, q), r in scalar_tables.items() if q == query]
        )
        for family in range(8, 16):
            row = natural_tables[family, query]
            actual = scalar_tables[family, query]["values"]
            anchor = row["values"][0]
            arms = {
                "constant_anchor": np.full(32, actual[0]),
                "logit_main": N.score(O.anchored_completion(anchor, c, MAIN), values),
                "logit_owner_binding": N.score(
                    O.anchored_completion(anchor, c, BINDING), values
                ),
                "logit_full_law": N.score(
                    O.anchored_completion(anchor, c, range(1, 32)), values
                ),
                "logit_wrong_binding": N.score(
                    O.anchored_completion(anchor, c, BINDING, wrong_binding=True),
                    values,
                ),
                "scalar_owner_binding": O.anchored_completion(actual[0], cs, BINDING),
            }
            for arm, predicted in arms.items():
                for mask in range(1, 32):
                    predictions.append(
                        {
                            "family": family,
                            "role": row["role"],
                            "query": query,
                            "mask": mask,
                            "arm": arm,
                            "prediction": float(predicted[mask]),
                            "actual": float(actual[mask]),
                            "squared_error": float(
                                (predicted[mask] - actual[mask]) ** 2
                            ),
                        }
                    )
        for role in ("validation", "application"):
            for arm in spec["arms"]:
                selected = [
                    r
                    for r in predictions
                    if r["query"] == query and r["role"] == role and r["arm"] == arm
                ]
                completion.append(
                    {
                        "query": query,
                        "role": role,
                        "arm": arm,
                        "cells": len(selected),
                        "families": len({r["family"] for r in selected}),
                        "rmse": float(
                            np.sqrt(np.mean([r["squared_error"] for r in selected]))
                        ),
                    }
                )
        interval = family_interval(
            np.array([natural_tables[f, query]["coefficients"] for f in range(8)]),
            [natural_tables[f, query]["values"] for f in range(12, 16)],
            [scalar_tables[f, query]["values"] for f in range(12, 16)],
            values,
            spec["seed"] + qi,
            spec["bootstrap_draws"],
        )
        intervals.append(
            {
                "query": query,
                "baseline": "logit_main",
                "arm": "logit_owner_binding",
                "source_families": 8,
                "application_families": 4,
                "draws": spec["bootstrap_draws"],
                "ci_low": interval[0],
                "ci_high": interval[1],
            }
        )
        for axis in ("anxiety", "trust"):
            selected = [
                r for r in bindings if r["query"] == query and r["axis"] == axis
            ]
            binding_summary.append(
                {
                    "query": query,
                    "axis": axis,
                    **{
                        name: float(np.mean([r[name] for r in selected]))
                        for name in (
                            "self_polarity_delta",
                            "other_polarity_delta",
                            "binding_contrast",
                            "affine_binding",
                            "natural_binding_norm",
                        )
                    },
                    "positive_binding_families": sum(
                        r["binding_contrast"] > 0 for r in selected
                    ),
                    "families": len(selected),
                }
            )
        for status in sorted(
            {r["evidence_status"] for r in readouts if r["query"] == query}
        ):
            selected = [
                r
                for r in readouts
                if r["query"] == query and r["evidence_status"] == status
            ]
            y = np.array([r["native"] for r in selected])
            evidence_summary.append(
                {
                    "query": query,
                    "evidence_status": status,
                    "records": len(y),
                    "unique_texts": len({r["text_id"] for r in selected}),
                    "native_min": float(y.min()),
                    "native_max": float(y.max()),
                    "native_mean": float(y.mean()),
                }
            )
    outputs = {
        "02/readouts.csv": readouts,
        "02/characters.csv": coefficients,
        "02/binding.csv": bindings,
        "02/conditional_link_split.csv": effects,
        "02/same_word_edges.csv": edges,
        "02/exact_raw_nulls.csv": controls,
        "03/predictions.csv": predictions,
        "03/completion_summary.csv": completion,
        "03/source_application_uncertainty.csv": intervals,
        "03/binding_summary.csv": binding_summary,
        "03/evidence_status_summary.csv": evidence_summary,
    }
    for name, rows in outputs.items():
        write_csv(out / name, rows)
    summary = {
        "packet": "HM-10B2",
        "status": "FINITE_TARGET_ORBITS_COMPLETE",
        "records": len(docs),
        "unique_texts": spec["unique_texts"],
        "actual_forwards": read_json(out / "run_status.json")["calls"],
        "full_tables": len(tables),
        "same_word_edges": len(edges),
        "exact_raw_nulls": len(controls),
        "character_rows": len(coefficients),
        "conditional_binding_tables": len(effects),
        "max_reconstruction_error": max_reconstruction,
        "max_energy_closure_error": max_energy,
        "max_binding_identity_error": max_interaction,
        "max_response_integral_error": max_integral,
        "binding_summary": binding_summary,
        "application_completion": [r for r in completion if r["role"] == "application"],
        "source_application_intervals": intervals,
        "evidence_status_summary": evidence_summary,
        "psychological_truth_supplied": False,
        "personal_response_parameters_fitted": False,
        "qualification_from_material_design": True,
        "new_neural_training": False,
        "outputs": {name: digest(out / name) for name in outputs},
    }
    write_json(out / "03/summary.json", summary)
    artifacts = {
        str(p.relative_to(out)): digest(p)
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    }
    write_json(out / "03/artifact_hashes.json", artifacts)
    print(json.dumps(summary), flush=True)


def main():
    """Run a finite freeze/collect/evaluate phase with a resumable collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if (
        not args.out.resolve().is_relative_to(
            (ROOT / "results/grain_math_response_v1").resolve()
        )
        or not 0 < args.seconds <= 1800
    ):
        parser.error("bounded local research output required")
    with collector_guard(args.out):
        {
            "prepare": lambda: prepare(args.out),
            "collect": lambda: collect(args.out, args.seconds),
            "evaluate": lambda: evaluate(args.out),
        }[args.mode]()


if __name__ == "__main__":
    main()

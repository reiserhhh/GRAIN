#!/usr/bin/env python3
"""HM10B3: bounded, restartable local owner-binding interventions."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_span_binding as B
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_span_binding_v1.json"
PLAN = ROOT / "docs/GRAIN_LOCAL_OWNER_BINDING_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/span_binding_r1"
CODE = (
    "scripts/run_grain_math_span_binding.py",
    "grain_research/construction/math_span_binding.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/counseling_queries.py",
    "grain_research/vector_mechanisms/context_carrier.py",
    "grain_research/vector_mechanisms/navigation.py",
    "scripts/run_grain_narrow_functions.py",
    "scripts/run_grain_math_native_joint.py",
    "scripts/run_grain_math_response.py",
    "scripts/run_grain_upstream_routes.py",
)


def intervention_design(pairs, spec):
    """Enumerate every intervention before observing any causal readout."""
    rows = []
    for family in sorted({p["family"] for p in pairs}):
        selected = [p for p in pairs if p["family"] == family]
        for p in selected:
            for layer in spec["layers"]:
                for arm in spec["arms"]:
                    for query in spec["queries"]:
                        rows.append(
                            {
                                **p,
                                "unit": len(rows),
                                "layer": layer,
                                "arm": arm,
                                "query": query,
                            }
                        )
        for p in selected:
            if p["polarity"] == 1 and p["order"] == 0:
                for query in spec["queries"]:
                    rows.append(
                        {
                            **p,
                            "unit": len(rows),
                            "layer": spec["noop_layer"],
                            "arm": "noop",
                            "query": query,
                        }
                    )
    return rows


def prepare(out):
    """Check old bytes, token identities and causal prefixes, then freeze the plan."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError(
            "existing prepared packets must be resumed, not overwritten"
        )
    spec = read_json(SPEC)
    previous = ROOT / spec["previous"]
    old = frozen_files(previous)
    hashes = read_json(previous / "03/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(previous / name) != expected:
            raise ValueError(f"old artifact changed: {name}")
    model_config = read_json(MODEL / "config.json")
    if max(spec["layers"]) >= model_config["num_hidden_layers"]:
        raise ValueError("requested decoder block does not exist")
    docs = [
        dict(d)
        for d in read_json(previous / "00/documents.json")
        if d["kind"] == "full" and d["mask"] in B.selected_masks()
    ]
    definitions = read_json(previous / "00/queries.json")
    old_units = {
        (d["text_id"], d["query"]): d for d in read_json(previous / "00/design.json")
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design, addresses = [], {}
    for doc_id, doc in enumerate(docs):
        doc["doc_id"] = doc_id
        addresses[doc["family"], doc["mask"]] = doc_id
        raw = B.raw_spans(doc["text"])
        first = None
        for query in spec["queries"]:
            rendered, body, encoded, mask = reference_view(
                tokenizer, [doc["text"]], definitions[query]["question"]
            )
            question_start = int(np.flatnonzero(mask)[0])
            ids = encoded["input_ids"]
            if (
                ids != old_units[doc["text_id"], query]["input_ids"]
                or np.asarray(ids)[mask].tolist() != old["question_tokens"][query]
            ):
                raise ValueError("old token/query coordinates changed")
            spans = {
                name: B.token_span(rendered, body[0][0], span, encoded, question_start)
                for name, span in raw.items()
            }
            common = (ids[:question_start], spans)
            if first is not None and common != first:
                raise ValueError("body prefix or span addresses depend on future query")
            first = common
            design.append(
                {
                    "unit": len(design),
                    "doc_id": doc_id,
                    "query": query,
                    "input_ids": ids,
                    "question_start": question_start,
                    "spans": spans,
                }
            )
        doc["spans"] = first[1]
    pairs = []
    for family in range(16):
        for p in B.owner_pairs():
            receiver, donor = (
                addresses[family, p["receiver_mask"]],
                addresses[family, p["donor_mask"]],
            )
            for suffix in ("property", "neutral"):
                name = f"{p['axis']}_{suffix}"
                x, y = docs[receiver]["spans"][name], docs[donor]["spans"][name]
                if x["raw"] != y["raw"] or x["token_ids"] != y["token_ids"]:
                    raise ValueError(
                        "donor/receiver property bytes or tokens do not align"
                    )
            pairs.append(
                {
                    **p,
                    "pair_id": len(pairs),
                    "family": family,
                    "role": docs[receiver]["role"],
                    "receiver_doc": receiver,
                    "donor_doc": donor,
                }
            )
    patches = intervention_design(pairs, spec)
    if (len(docs), len(design), len(pairs), len(patches)) != (
        spec["document_records"],
        spec["capture_forwards"],
        spec["owner_pairs"],
        spec["patch_forwards"],
    ):
        raise ValueError("coverage differs from declared design")
    counts = [
        len({d["family"] for d in docs if d["role"] == role})
        for role in ("source", "validation", "application")
    ]
    if counts != spec["source_validation_application_families"]:
        raise ValueError("source/application family roles changed")
    sources = {
        name: digest(ROOT / name)
        for name in (
            *CODE,
            str(SPEC.relative_to(ROOT)),
            str(PLAN.relative_to(ROOT)),
            str((previous / "03/artifact_hashes.json").relative_to(ROOT)),
            str((previous / "02/readouts.csv").relative_to(ROOT)),
            str((previous / "00/manifest.json").relative_to(ROOT)),
            str((MODEL / "config.json").relative_to(ROOT)),
        )
    }
    inputs = {
        "documents": docs,
        "design": design,
        "pairs": pairs,
        "patch_design": patches,
        "queries": {q: definitions[q] for q in spec["queries"]},
    }
    (out / "01/units").mkdir(parents=True)
    (out / "03/units").mkdir(parents=True)
    for name, value in inputs.items():
        write_json(out / f"00/{name}.json", value)
    for name in CODE:
        target = out / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "packet": spec["packet"],
            "spec": spec,
            "alphabet": old["alphabet"],
            "model_sha256": old["model_sha256"],
            "inputs": {
                f"00/{name}.json": digest(out / f"00/{name}.json") for name in inputs
            },
            "source_files": sources,
            "previous_hashes_checked": len(hashes),
            "previous_source_files_checked": len(old["source_files"]),
            "model_depth": model_config["num_hidden_layers"],
            "hidden_dimension": model_config["hidden_size"],
            "frozen_before_new_calls": True,
            "body_prefix_same_across_queries": True,
            "new_raw_texts": 0,
            "api_calls": 0,
            "private_meps_reads": 0,
            "new_neural_training": False,
            "psychological_truth": False,
            "local_total_effect_not_storage_only_mediation": True,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "capture_calls": 0,
            "patch_calls": 0,
            "capture_units": 0,
            "patch_units": 0,
            "seconds": 0,
        },
    )
    print(
        {
            "status": "PREPARED",
            "documents": len(docs),
            "captures": len(design),
            "pairs": len(pairs),
            "patches": len(patches),
        },
        flush=True,
    )


def array_unit(path):
    """Validate an atomic captured unit and its array before reuse."""
    meta = read_json(path)
    if digest(path.with_suffix(".npz")) != meta["array_sha256"]:
        raise ValueError(f"captured array changed: {path}")
    return meta


def save_array_unit(path, metadata, arrays):
    """Save arrays first and metadata last, so partial writes cannot count as units."""
    target = path.with_suffix(".npz")
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(target)
    write_json(path, {**metadata, "array_sha256": digest(target)})


def load_observer():
    """Use the same local frozen float32 eager reader, without generation or fitting."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    return observer


def forward(observer, ids, alphabet):
    """Run one actual fixed-template forward and record full-vocabulary compliance."""
    import torch

    tokens = torch.tensor(
        [ids], dtype=torch.long, device=next(observer.model.parameters()).device
    )
    with torch.inference_mode():
        output = observer.model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    observer.calls += 1
    observer.input_tokens += len(ids)
    observer.processed_tokens += len(ids)
    return native_readout(output.logits[0, -1].float().cpu().numpy(), alphabet)


def hidden_tensor(output):
    """Support tensor and tuple decoder outputs without confusing block indices."""
    return output[0] if isinstance(output, tuple) else output


def capture(out):
    """Collect selected span states; check old readouts and future-query independence."""
    manifest = frozen_files(out)
    previous = read_json(out / "run_status.json")
    design = read_json(out / "00/design.json")
    if all((out / "01/units" / f"unit_{r['unit']:04d}.json").exists() for r in design):
        print({"status": "CAPTURE_ALREADY_COMPLETE", "units": len(design)}, flush=True)
        return
    spec, observer = manifest["spec"], load_observer()
    write_json(out / "01/observer.json", observer.manifest())
    docs = read_json(out / "00/documents.json")
    with (ROOT / spec["previous"] / "02/readouts.csv").open() as stream:
        baseline = {
            (r["material_id"], r["query"]): float(r["native"])
            for r in csv.DictReader(stream)
        }
    canonical = {
        r["doc_id"]: r["unit"] for r in design if r["query"] == spec["queries"][0]
    }
    start, complete = time.monotonic(), 0

    def status(label):
        result = {
            **previous,
            "status": label,
            "capture_units": complete,
            "capture_calls": previous["capture_calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", result)
        return result

    status("CAPTURING")
    try:
        for row in design:
            path = out / "01/units" / f"unit_{row['unit']:04d}.json"
            if path.exists():
                saved = array_unit(path)
                if saved["address"] != row:
                    raise ValueError("saved capture address changed")
                complete += 1
                continue
            if (
                observer.calls >= spec["max_stage_calls"]
                or time.monotonic() - start >= spec["max_stage_seconds"]
            ):
                print(status("CAPTURE_RESOURCE_BOUNDARY"), flush=True)
                return
            captured, handles = {}, []
            for layer in spec["layers"]:

                def hook(
                    _module, _args, output, layer=layer, row=row, captured=captured
                ):
                    h = hidden_tensor(output)[0]
                    for name, span in row["spans"].items():
                        captured[f"L{layer}_{name}"] = (
                            h[span["positions"]].detach().float().cpu().numpy().copy()
                        )
                    captured[f"L{layer}_Q"] = (
                        h[-1].detach().float().cpu().numpy().copy()
                    )

                handles.append(
                    observer.model.model.layers[layer].register_forward_hook(hook)
                )
            try:
                native = forward(observer, row["input_ids"], manifest["alphabet"])
            finally:
                for handle in handles:
                    handle.remove()
            old = baseline[docs[row["doc_id"]]["material_id"], row["query"]]
            score_error = abs(native["conditional_score"] - old)
            canonical_path = (
                out / "01/units" / f"unit_{canonical[row['doc_id']]:04d}.npz"
            )
            body_error, scale = 0.0, 0.0
            if row["query"] != spec["queries"][0]:
                with np.load(canonical_path, allow_pickle=False) as arrays:
                    for key, value in captured.items():
                        if not key.endswith("_Q"):
                            body_error = max(
                                body_error,
                                float(
                                    np.max(
                                        np.abs(
                                            value.astype(float)
                                            - arrays[key].astype(float)
                                        )
                                    )
                                ),
                            )
                            scale = max(scale, float(np.max(np.abs(arrays[key]))))
                saved_arrays = {k: v for k, v in captured.items() if k.endswith("_Q")}
            else:
                saved_arrays = captured
            meta = {
                "address": row,
                "native": native,
                "old_native": old,
                "old_readout_error": score_error,
                "body_query_error": body_error,
                "body_query_scale": scale,
                "canonical_capture_unit": canonical[row["doc_id"]],
            }
            save_array_unit(path, meta, saved_arrays)
            complete += 1
            if (
                score_error > spec["baseline_readout_tolerance"]
                or body_error
                > spec["body_query_absolute_tolerance"]
                + spec["body_query_relative_tolerance"] * scale
            ):
                raise ValueError(
                    "old baseline or causal body-prefix replay exceeds fixed tolerance; failed unit preserved"
                )
            if complete % 24 == 0:
                print(status("CAPTURING"), flush=True)
        print(status("COMPLETE_CAPTURE"), flush=True)
    except BaseException as error:
        result = status("CAPTURE_FAILED")
        write_json(out / "capture_failure.json", {**result, "error": repr(error)})
        raise


def build_basis(out):
    """Freeze two source-only mean directions per layer and a fixed random frame."""
    manifest = frozen_files(out)
    if (out / "02/basis_manifest.json").exists():
        checked_basis(out)
        print({"status": "BASIS_ALREADY_FROZEN"}, flush=True)
        return
    spec, design = manifest["spec"], read_json(out / "00/design.json")
    meta = [array_unit(out / "01/units" / f"unit_{d['unit']:04d}.json") for d in design]
    if any(
        m["old_readout_error"] > spec["baseline_readout_tolerance"]
        or m["body_query_error"]
        > spec["body_query_absolute_tolerance"]
        + spec["body_query_relative_tolerance"] * m["body_query_scale"]
        for m in meta
    ):
        raise ValueError("capture replay failed; no frame or patch accepted")
    canonical = {
        d["doc_id"]: d["unit"] for d in design if d["query"] == spec["queries"][0]
    }
    source_pairs = [
        p for p in read_json(out / "00/pairs.json") if p["role"] == "source"
    ]
    bodies = {}
    for doc_id in {
        p[key] for p in source_pairs for key in ("receiver_doc", "donor_doc")
    }:
        with np.load(
            out / "01/units" / f"unit_{canonical[doc_id]:04d}.npz", allow_pickle=False
        ) as a:
            bodies[doc_id] = {
                k: a[k].astype(float) for k in a.files if not k.endswith("_Q")
            }
    frames, rows = {}, []
    for layer in spec["layers"]:
        contrasts = {
            axis: [
                (
                    bodies[p["donor_doc"]][f"L{layer}_{axis}_property"]
                    - bodies[p["receiver_doc"]][f"L{layer}_{axis}_property"]
                ).mean(axis=0)
                for p in source_pairs
                if p["axis"] == axis
            ]
            for axis in ("A", "T")
        }
        if [len(contrasts[a]) for a in ("A", "T")] != [32, 32]:
            raise ValueError("source direction count changed")
        u, directions, singular = B.shared_basis(contrasts["A"], contrasts["T"])
        r = B.random_basis(u.shape[0], u.shape[1], spec["random_seed"] + layer)
        for name, value in (
            ("U", u),
            ("D", directions),
            ("singular", singular),
            ("R", r),
        ):
            frames[f"L{layer}_{name}"] = value
        rows.append(
            {
                "layer": layer,
                "rank": u.shape[1],
                "source_pairs_per_axis": 32,
                "singular_values": singular.tolist(),
                "direction_cosine": float(
                    directions[:, 0]
                    @ directions[:, 1]
                    / np.prod(np.linalg.norm(directions, axis=0))
                ),
                "orthogonality_error": float(
                    np.max(np.abs(u.T @ u - np.eye(u.shape[1])))
                ),
                "direction_reconstruction_error": float(
                    np.max(np.abs((directions.T @ u) @ u.T - directions.T))
                ),
            }
        )
    (out / "02").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "02/basis.npz", **frames)
    capture_hashes = {
        str(path.relative_to(out)): digest(path)
        for path in sorted((out / "01/units").glob("unit_*"))
    }
    write_json(
        out / "02/basis_manifest.json",
        {
            "array_sha256": digest(out / "02/basis.npz"),
            "capture_hashes": capture_hashes,
            "source_pair_ids": [p["pair_id"] for p in source_pairs],
            "source_only": True,
            "query_independent_frame": True,
            "patch_calls_before_freeze": read_json(out / "run_status.json")[
                "patch_calls"
            ],
            "layers": rows,
        },
    )
    write_csv(out / "02/frame_geometry.csv", rows)
    write_json(
        out / "run_status.json",
        {**read_json(out / "run_status.json"), "status": "BASIS_FROZEN"},
    )
    print({"status": "BASIS_FROZEN", "layers": rows}, flush=True)


def checked_basis(out):
    """Refuse changed frame/capture bytes or a frame learned after patches."""
    m = read_json(out / "02/basis_manifest.json")
    if (
        m["patch_calls_before_freeze"] != 0
        or not m["source_only"]
        or digest(out / "02/basis.npz") != m["array_sha256"]
    ):
        raise ValueError("binding frame was changed or learned after intervention")
    for name, expected in m["capture_hashes"].items():
        if digest(out / name) != expected:
            raise ValueError("frozen capture changed")
    with np.load(out / "02/basis.npz", allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def checked_patch(path, row, tolerance):
    """Reject changed addresses, arithmetic or failed no-ops on every resume."""
    record = read_json(path)
    if record["address"] != row:
        raise ValueError("saved intervention address changed")
    actual = record["native"]["conditional_score"] - record["receiver_native"]
    if not np.isfinite(actual) or abs(actual - record["patch_delta"]) > 1e-12:
        raise ValueError("saved intervention arithmetic changed")
    if row["arm"] == "noop" and abs(actual) > tolerance:
        raise ValueError("saved no-op replay failed")
    return record


def patch(out):
    """Patch actual local states and recompute the receiver's downstream network."""
    import torch

    manifest = frozen_files(out)
    frames = checked_basis(out)
    previous = read_json(out / "run_status.json")
    spec = manifest["spec"]
    design = read_json(out / "00/patch_design.json")
    if all((out / "03/units" / f"unit_{r['unit']:05d}.json").exists() for r in design):
        for row in design:
            checked_patch(
                out / "03/units" / f"unit_{row['unit']:05d}.json",
                row,
                spec["baseline_readout_tolerance"],
            )
        print({"status": "PATCH_ALREADY_COMPLETE", "units": len(design)}, flush=True)
        return
    inputs = {(d["doc_id"], d["query"]): d for d in read_json(out / "00/design.json")}
    baseline = {
        k: array_unit(out / "01/units" / f"unit_{r['unit']:04d}.json")["native"][
            "conditional_score"
        ]
        for k, r in inputs.items()
    }
    canonical = {
        d["doc_id"]: d["unit"]
        for d in inputs.values()
        if d["query"] == spec["queries"][0]
    }
    bodies = {}
    for doc_id, unit in canonical.items():
        with np.load(
            out / "01/units" / f"unit_{unit:04d}.npz", allow_pickle=False
        ) as a:
            bodies[doc_id] = {k: a[k] for k in a.files if not k.endswith("_Q")}
    observer = load_observer()
    write_json(out / "03/observer.json", observer.manifest())
    start, complete = time.monotonic(), 0

    def status(label):
        result = {
            **previous,
            "status": label,
            "patch_units": complete,
            "patch_calls": previous["patch_calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", result)
        return result

    status("PATCHING")
    try:
        for row in design:
            path = out / "03/units" / f"unit_{row['unit']:05d}.json"
            if path.exists():
                checked_patch(path, row, spec["baseline_readout_tolerance"])
                complete += 1
                continue
            if (
                observer.calls >= spec["max_stage_calls"]
                or time.monotonic() - start >= spec["max_stage_seconds"]
            ):
                print(status("PATCH_RESOURCE_BOUNDARY"), flush=True)
                return
            receiver = inputs[row["receiver_doc"], row["query"]]
            name = (
                f"{row['axis']}_{'neutral' if row['arm'] == 'neutral' else 'property'}"
            )
            positions = receiver["spans"][name]["positions"]
            key = f"L{row['layer']}_{name}"
            donor = bodies[row["donor_doc"]][key].astype(float)
            u, r = frames[f"L{row['layer']}_U"], frames[f"L{row['layer']}_R"]
            audit = {}

            def hook(
                _module,
                _args,
                output,
                positions=positions,
                donor=donor,
                u=u,
                r=r,
                row=row,
                key=key,
                audit=audit,
            ):
                h = hidden_tensor(output)
                current = h[0, positions].detach().float().cpu().numpy().astype(float)
                delta = B.intervention_delta(current, donor, row["arm"], u, r)
                projection = B.intervention_delta(current, donor, "projection", u, r)
                complement = B.intervention_delta(current, donor, "complement", u, r)
                audit.update(
                    {
                        "patched_tokens": len(positions),
                        "donor_delta_norm": float(np.linalg.norm(donor - current)),
                        "projection_norm": float(np.linalg.norm(projection)),
                        "intervention_norm": float(np.linalg.norm(delta)),
                        "split_error": float(
                            np.max(np.abs(projection + complement - (donor - current)))
                        ),
                        "live_body_error": float(
                            np.max(
                                np.abs(
                                    current
                                    - bodies[row["receiver_doc"]][key].astype(float)
                                )
                            )
                        ),
                    }
                )
                if row["arm"] == "noop":
                    return output
                altered = h.clone()
                target = (
                    donor
                    if row["arm"] in ("full_property", "neutral")
                    else current + delta
                )
                altered[0, positions] = torch.as_tensor(
                    target, dtype=h.dtype, device=h.device
                )
                return (altered, *output[1:]) if isinstance(output, tuple) else altered

            handle = observer.model.model.layers[row["layer"]].register_forward_hook(
                hook
            )
            try:
                native = forward(observer, receiver["input_ids"], manifest["alphabet"])
            finally:
                handle.remove()
            if not audit:
                raise ValueError("declared decoder hook was not called")
            base = baseline[row["receiver_doc"], row["query"]]
            target_score = baseline[row["donor_doc"], row["query"]]
            record = {
                "address": row,
                "native": native,
                "receiver_native": base,
                "donor_native": target_score,
                "patch_delta": native["conditional_score"] - base,
                "natural_owner_delta": target_score - base,
                **audit,
            }
            write_json(path, record)
            complete += 1
            if (
                row["arm"] == "noop"
                and abs(record["patch_delta"]) > spec["baseline_readout_tolerance"]
            ):
                raise ValueError(
                    "no-op replay failed; intervention saved but not accepted"
                )
            if complete % 100 == 0:
                print(status("PATCHING"), flush=True)
        print(status("COMPLETE_PATCH"), flush=True)
    except BaseException as error:
        result = status("PATCH_FAILED")
        write_json(out / "patch_failure.json", {**result, "error": repr(error)})
        raise


def evaluate(out):
    """Report every completed arm without promoting a partial family as complete."""
    manifest = frozen_files(out)
    frames = checked_basis(out)
    design = read_json(out / "00/patch_design.json")
    records, grouped = [], defaultdict(list)
    for row in design:
        path = out / "03/units" / f"unit_{row['unit']:05d}.json"
        if not path.exists():
            continue
        record = checked_patch(
            path, row, manifest["spec"]["baseline_readout_tolerance"]
        )
        flat = {
            **row,
            **{k: v for k, v in record.items() if k not in ("address", "native")},
            "patched_native": record["native"]["conditional_score"],
            "candidate_mass": record["native"]["candidate_mass"],
        }
        records.append(flat)
        if row["arm"] != "noop":
            grouped[
                row["family"],
                row["axis"],
                row["order"],
                row["layer"],
                row["arm"],
                row["query"],
            ].append(flat)
    contrasts = []
    for key, rows in grouped.items():
        if {r["polarity"] for r in rows} != {0, 1} or len(rows) != 2:
            continue
        high, low = sorted(rows, key=lambda r: r["polarity"], reverse=True)
        contrasts.append(
            {
                "family": key[0],
                "role": high["role"],
                "axis": key[1],
                "order": key[2],
                "layer": key[3],
                "arm": key[4],
                "query": key[5],
                "binding_action": high["patch_delta"] - low["patch_delta"],
                "natural_binding_action": high["natural_owner_delta"]
                - low["natural_owner_delta"],
                "high_delta": high["patch_delta"],
                "low_delta": low["patch_delta"],
            }
        )
    summaries = []
    blocks = defaultdict(list)
    for row in contrasts:
        blocks[
            row["role"],
            row["axis"],
            row["layer"],
            row["arm"],
            row["query"],
            row["order"],
        ].append(row)
    for key, rows in blocks.items():
        summaries.append(
            {
                "role": key[0],
                "axis": key[1],
                "layer": key[2],
                "arm": key[3],
                "query": key[4],
                "order": key[5],
                "families": len(rows),
                "mean_binding_action": float(
                    np.mean([r["binding_action"] for r in rows])
                ),
                "mean_natural_binding_action": float(
                    np.mean([r["natural_binding_action"] for r in rows])
                ),
            }
        )
    noop = [r for r in records if r["arm"] == "noop"]
    geometry = []
    for layer in manifest["spec"]["layers"]:
        u, r = frames[f"L{layer}_U"], frames[f"L{layer}_R"]
        geometry.append(
            {
                "layer": layer,
                "rank": u.shape[1],
                "frame_overlap_energy": float(np.sum((u.T @ r) ** 2)),
            }
        )
    status = read_json(out / "run_status.json")
    summary = {
        "status": "COMPLETE" if len(records) == len(design) else "PARTIAL_NO_PROMOTION",
        "expected_units": len(design),
        "completed_units": len(records),
        "complete_polarity_contrasts": len(contrasts),
        "capture_calls": status["capture_calls"],
        "patch_calls": status["patch_calls"],
        "noop_units": len(noop),
        "noop_max_error": max((abs(r["patch_delta"]) for r in noop), default=None),
        "split_max_error": max((r["split_error"] for r in records), default=None),
        "raw_text_replication": False,
        "human_personality_claim": False,
        "all_summary_cells": summaries,
        "random_frame_geometry": geometry,
    }
    write_csv(out / "04/interventions.csv", records)
    write_csv(out / "04/binding_action_by_family.csv", contrasts)
    write_csv(out / "04/binding_summary.csv", summaries)
    write_json(out / "04/summary.json", summary)
    print(
        {
            k: v
            for k, v in summary.items()
            if k not in ("all_summary_cells", "random_frame_geometry")
        },
        flush=True,
    )


def main():
    """One bounded stage per invocation; resource boundaries are resumable pauses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "capture", "basis", "patch", "evaluate")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with collector_guard(args.out):
        {
            "prepare": prepare,
            "capture": capture,
            "basis": build_basis,
            "patch": patch,
            "evaluate": evaluate,
        }[args.stage](args.out)


if __name__ == "__main__":
    main()

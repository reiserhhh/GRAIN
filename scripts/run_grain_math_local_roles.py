#!/usr/bin/env python3
"""HM-04: cold heterogeneous event-role binding, with no new model fitting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

from grain_research.construction import math_binding as B
from grain_research.construction import math_local as L
from grain_research.construction import math_local_roles as R
from grain_research.source_event_graph import stanza_words

SPEC = ROOT / "configs/grain_math_local_roles_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/local_roles_r1"


def prepare(out):
    """Freeze new composition materials and every source/program dependency."""
    spec = json.loads(SPEC.read_text())
    source = ROOT / spec["source_packet"]
    hashes = json.loads((source / "04/artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(source / name) != digest:
            raise ValueError(f"source packet changed: {name}")
    originals = read_bank(source / "00/singles.jsonl")
    bank = R.material_bank(originals, spec["styles"])
    code = [
        Path(__file__),
        ROOT / "grain_research/construction/math_local_roles.py",
        SPEC,
        ROOT / "grain_research/construction/math_binding.py",
        ROOT / "grain_research/construction/math_local.py",
        ROOT / "grain_research/source_event_graph.py",
        ROOT / "scripts/run_grain_math_response.py",
        ROOT / "scripts/run_grain_math_binding.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in code}
    signature["source_artifact_manifest"] = sha(source / "04/artifact_hashes.json")
    protocol = out / "00/protocol_before_run.json"
    content = "".join(json.dumps(r) + "\n" for r in bank)
    if protocol.exists():
        old = json.loads(protocol.read_text())
        if (
            old["signature"] != signature
            or (out / "00/documents.jsonl").read_text() != content
        ):
            raise ValueError("HM-04 protocol changed: preserve r1 and use a new output")
        return source, originals, bank
    for p in code:
        dest = out / "00/code" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    (out / "00/documents.jsonl").write_text(content)
    write_json(
        out / "00/reader_registry.json",
        json.loads((source / "00/reader_registry.json").read_text()),
    )
    write_json(
        protocol,
        {
            "signature": signature,
            "spec": spec,
            "source_hashes_verified": len(hashes),
            "documents": len(bank),
            "known_scope_documents": sum(not r["ambiguous"] for r in bank),
            "ambiguous_scope_documents": sum(r["ambiguous"] for r in bank),
            "semantic_source": "HM-03 cold object components; new role compositions; no refit",
            "target": "reported local action expression, not verified actions or personality",
            "parser": "frozen Stanza; role projection rules set before new encoding",
            "aliases": "record relations supplied independently of role labels",
            "new_fits": 0,
            "new_neural_training": 0,
            "private_reads": 0,
            "api_calls": 0,
        },
    )
    write_json(
        out / "00/marginal_collision_witness.json", R.marginal_collision_witness()
    )
    print(f"HM-04 frozen: {len(bank)} documents; no new fits", flush=True)
    return source, originals, bank


def parse_materials(out, bank):
    """Run actual dependency parsing, cache raw words and addressed projections."""
    from fetch_grain_syntax_reference import MODELS as SYNTAX
    from fetch_grain_syntax_reference import PACKAGES, PROCESSORS
    from run_grain_source_event_graph import verify_models

    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    if stanza.__version__ != "1.11.0":
        raise ValueError("frozen parser version changed")
    torch.set_num_threads(2)
    parses, pipeline, calls, start = [], None, 0, time.monotonic()
    for i, row in enumerate(bank):
        path = out / f"01/parses/{i:04d}.json"
        digest = R.text_digest(row["text"])
        if path.exists():
            data = json.loads(path.read_text())
            if data["text_sha256"] != digest:
                raise ValueError("parser cache mismatch")
        else:
            if pipeline is None:
                pipeline = stanza.Pipeline(
                    "en",
                    dir=str(SYNTAX),
                    package=None,
                    processors=PROCESSORS,
                    use_gpu=False,
                    download_method=None,
                    verbose=False,
                )
            words = stanza_words(pipeline(row["text"]), row["text"])
            events = R.attach_event_identity(
                R.project_events(words, row["text"]), row["events"]
            )
            data = {"text_sha256": digest, "words": words, "events": events}
            write_json(path, data)
            calls += 1
        parses.append(data)
        if (i + 1) % 256 == 0:
            print(
                f"HM-04 parser {i + 1}/{len(bank)} elapsed={time.monotonic() - start:.1f}s",
                flush=True,
            )
    write_json(
        out / "01/parser_accounting.json",
        {
            "material_documents": len(bank),
            "new_calls": calls,
            "cache_rows": len(bank) - calls,
            "elapsed": time.monotonic() - start,
            "syntax_inventory_sha256": sha(SYNTAX / "grain_inventory.json"),
        },
    )
    return parses


def observation_inputs(out, source, bank, parses):
    """Reuse exact frozen component vectors and encode only new public inputs."""
    original_inputs = read_bank(source / "00/material_bank.jsonl")
    old = {r["text"]: i for i, r in enumerate(original_inputs)}
    fresh, lookup = [], {}

    def ref(text, kind):
        if text in old:
            return ["source", old[text]]
        if text not in lookup:
            lookup[text] = len(fresh)
            fresh.append({"text": text, "kind": kind})
        return ["new", lookup[text]]

    doc_refs, event_refs, alignment = [], [], []
    for row, parsed in zip(bank, parses, strict=True):
        doc_refs.append(ref(row["text"], "document"))
        observations = []
        for event in parsed["events"]:
            observations.append(
                {
                    "event_id": event["event_id"],
                    "record_index": event["record_index"],
                    "status": event["status"],
                    "readout": ref(event["projection_text"], "local_projection"),
                    "action": ref("Action: " + event["action"] + ".", "action_slot"),
                    "object": ref("Object: " + event["object"] + ".", "object_slot"),
                }
            )
            expected = [
                e
                for e in row["events"]
                if e["span"][0] <= event["action_span"][0] < e["span"][1]
            ]
            alignment.append(
                {
                    "material_id": row["material_id"],
                    "target": row["target"],
                    "style": row["style"],
                    "kind": row["kind"],
                    "status": event["status"],
                    "event_id": event["event_id"],
                    "flags": event["flags"],
                    "subject_origin": event["subject_origin"],
                    "auxiliary_origin": event["auxiliary_origin"],
                    "object_exact": len(expected) == 1
                    and event["object"]
                    == original_inputs[expected[0]["source_index"]]["object"],
                    "projection_text": event["projection_text"],
                }
            )
        event_refs.append(observations)
    content = "".join(json.dumps(r) + "\n" for r in fresh)
    path = out / "00/material_bank.jsonl"
    if path.exists() and path.read_text() != content:
        raise ValueError("observation bank changed; cache may not be reused")
    path.write_text(content)
    write_json(
        out / "01/observation_indices.json",
        {
            "documents": doc_refs,
            "automatic_events": event_refs,
            "encoder_inputs_per_reader": len(fresh),
            "reused_source_unique_inputs": len(old),
            "reuse_rule": "exact public string and identical frozen reader only",
        },
    )
    write_csv(out / "01/parser_alignment.csv", alignment)
    return doc_refs, event_refs


def vectors(refs, original, fresh):
    """Resolve numeric observations only; no labels or document IDs enter scoring."""
    if not refs:
        return np.empty((0, original.shape[1]))
    return np.array(
        [original[i] if space == "source" else fresh[i] for space, i in refs]
    )


def value_profile(ids, z, *, unresolved=0):
    """Evaluate the fixed event contribution and routing from observed coordinates."""
    scores, factors = B.product_scores(z)
    return R.event_profile(
        ids, scores, L.qualify(factors), unresolved=unresolved
    ), scores


def evaluate(out, source, originals, bank, doc_refs, event_refs):
    """Cold score all channels; evaluation contracts are consulted afterwards."""
    predictions, event_rows, replay_errors = [], [], []
    declared = np.array(
        json.loads((source / "01/slot_indices.json").read_text())["declared"]
    )
    for reader in MODELS:
        original = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        fresh = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        dx = vectors(doc_refs, original, fresh)
        for target in sorted({r["target"] for r in bank}):
            models = load_model(source / f"02/{reader}_{target}.npz")
            zdoc = L.affine_score(models["whole_factors"], dx)
            zdecl = L.affine_score(models["whole_factors"], original[: len(originals)])
            kdecl = B.score_kernel(
                models["slot_tensor"],
                original[declared[:, 0]],
                original[declared[:, 1]],
            )
            zdecl[:, 0] = kdecl
            flat = [
                e
                for i, r in enumerate(bank)
                if r["target"] == target
                for e in event_refs[i]
            ]
            ax = vectors([e["action"] for e in flat], original, fresh)
            bx = vectors([e["object"] for e in flat], original, fresh)
            zx = L.affine_score(
                models["whole_factors"],
                vectors([e["readout"] for e in flat], original, fresh),
            )
            zx[:, 0] = B.score_kernel(models["auto_tensor"], ax, bx)
            cursor = 0
            for i, row in enumerate(bank):
                if row["target"] != target:
                    continue
                supplied = [{"event_id": e["event_id"]} for e in row["events"]]
                si = [e["source_index"] for e in row["events"]]
                zg = np.tile(zdoc[i], (len(si), 1))
                zg[:, 0] = kdecl[si]
                zl = zdecl[si].copy()
                broken = zl.copy()
                broken[:, 2] = np.roll(broken[:, 2], 1)
                auto = event_refs[i]
                za = zx[cursor : cursor + len(auto)]
                cursor += len(auto)
                valid = np.array(
                    [e["status"] == "ok" and e["event_id"] is not None for e in auto]
                )
                missing = R.missing_records(auto, len(supplied))
                auto_ids = [
                    {"event_id": e["event_id"]}
                    for e, yes in zip(auto, valid, strict=True)
                    if yes
                ]
                zas = np.tile(zdoc[i], (len(za), 1))
                zas[:, 0] = za[:, 0]
                arms = {
                    "whole_document": ([{"event_id": "document"}], zdoc[i : i + 1], 0),
                    "declared_shared": (supplied, zg, 0),
                    "declared_local": (supplied, zl, 0),
                    "broken_ownership_binding": (supplied, broken, 0),
                    "automatic_shared": (auto_ids, zas[valid], missing),
                    "automatic_local": (auto_ids, za[valid], missing),
                }
                truth = R.true_profile(row)
                for arm, (ids, z, unresolved) in arms.items():
                    profile, sc = value_profile(ids, z, unresolved=unresolved)
                    scalar = (
                        z[:, 1]
                        * np.clip((1 + z[:, 0]) / 2, 0, 1)
                        * np.clip((1 + z[:, 2]) / 2, 0, 1)
                        * np.clip((1 + z[:, 3]) / 2, 0, 1)
                    )
                    if len(sc):
                        replay_errors.append(float(np.max(abs(sc - scalar))))
                    predictions.append(
                        {
                            "reader": reader,
                            "target": target,
                            "family": row["family"],
                            "style": row["style"],
                            "kind": row["kind"],
                            "material_id": row["material_id"],
                            "arm": arm,
                            "ambiguous": row["ambiguous"],
                            "prediction": profile["score"],
                            "unit": profile["unit"],
                            "present": bool(profile["observed_events"])
                            if profile["complete"]
                            else None,
                            "complete": profile["complete"],
                            "unresolved": profile["unresolved"],
                            "observed_events": profile["observed_events"],
                            "unique_events": profile["unique_events"],
                            "alias_score_variation": profile["alias_score_variation"],
                            "expected_score": truth["score"],
                            "expected_unit": truth["unit"],
                            "expected_present": bool(truth["observed_events"]),
                            "expected_events": truth["observed_events"],
                        }
                    )
                    for e, factors, value in zip(ids, z, sc, strict=True):
                        event_rows.append(
                            {
                                "reader": reader,
                                "target": target,
                                "material_id": row["material_id"],
                                "arm": arm,
                                "event_id": e["event_id"],
                                "k": float(factors[0]),
                                "p": float(factors[1]),
                                "o": float(factors[2]),
                                "a": float(factors[3]),
                                "contribution": float(value),
                            }
                        )
    error = max(replay_errors, default=0)
    if error > 1e-12:
        raise AssertionError(error)
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/event_readouts.csv", event_rows)
    write_json(
        out / "04/scalar_replay.json",
        {
            "status": "PASS",
            "max_error": error,
            "event_arrays": len(replay_errors),
            "new_fits": 0,
        },
    )
    return predictions


def finite_metrics(out, bank, predictions):
    """Separate observability, contribution error, counterfactuals and aliases."""
    from collections import defaultdict

    groups, metrics = defaultdict(list), []
    lookup = {(r["reader"], r["arm"], r["material_id"]): r for r in predictions}
    for p in predictions:
        if not p["ambiguous"]:
            groups[p["reader"], p["target"], p["style"], p["kind"], p["arm"]].append(p)
    for key, rows in groups.items():
        good = [p for p in rows if p["complete"]]
        scored = [
            p
            for p in good
            if p["prediction"] is not None and p["expected_score"] is not None
        ]
        units = [p for p in good if p["unit"] is not None]
        metrics.append(
            {
                **dict(
                    zip(("reader", "target", "style", "kind", "arm"), key, strict=True)
                ),
                "n": len(rows),
                "complete": len(good),
                "coverage": len(good) / len(rows),
                "unit_rmse": float(
                    np.sqrt(
                        np.mean([(p["unit"] - p["expected_unit"]) ** 2 for p in units])
                    )
                )
                if units
                else None,
                "joint_observed_n": len(scored),
                "profile_rmse_joint_observed": float(
                    np.sqrt(
                        np.mean(
                            [
                                (p["prediction"] - p["expected_score"]) ** 2
                                for p in scored
                            ]
                        )
                    )
                )
                if scored
                else None,
                "tp": sum(p["present"] and p["expected_present"] for p in good),
                "fp": sum(p["present"] and not p["expected_present"] for p in good),
                "fn": sum(not p["present"] and p["expected_present"] for p in good),
                "tn": sum(not p["present"] and not p["expected_present"] for p in good),
            }
        )
    by_roles = {
        (r["target"], r["family"], r["style"], tuple(r["roles"])): r
        for r in bank
        if r["kind"] == "mixture"
    }
    contrasts, alias_rows, comparisons = [], [], []
    for row in bank:
        if row["kind"] == "mixture":
            for j, factor in enumerate(("polarity", "ownership", "actuality")):
                roles = row["roles"].copy()
                roles[j], roles[j + 3] = roles[j + 3], roles[j]
                if roles == row["roles"]:
                    continue
                alt = by_roles[row["target"], row["family"], row["style"], tuple(roles)]
                if row["material_id"] >= alt["material_id"]:
                    continue
                left, right = R.true_profile(row), R.true_profile(alt)
                if left["unit"] == right["unit"]:
                    continue
                if sorted(
                    row["text"].lower().replace(",", "").replace(".", "").split()
                ) != sorted(
                    alt["text"].lower().replace(",", "").replace(".", "").split()
                ):
                    raise AssertionError("counterfactual word bags differ")
                for reader in MODELS:
                    for arm in (
                        "whole_document",
                        "declared_shared",
                        "declared_local",
                        "automatic_shared",
                        "automatic_local",
                        "broken_ownership_binding",
                    ):
                        x, y = (
                            lookup[reader, arm, row["material_id"]],
                            lookup[reader, arm, alt["material_id"]],
                        )
                        delta = (
                            y["unit"] - x["unit"]
                            if x["complete"]
                            and y["complete"]
                            and x["unit"] is not None
                            and y["unit"] is not None
                            else None
                        )
                        contrasts.append(
                            {
                                "reader": reader,
                                "target": row["target"],
                                "style": row["style"],
                                "family": row["family"],
                                "factor": factor,
                                "arm": arm,
                                "left_id": row["material_id"],
                                "right_id": alt["material_id"],
                                "expected_delta": right["unit"] - left["unit"],
                                "predicted_delta": delta,
                                "correct_direction": (
                                    delta * (right["unit"] - left["unit"]) > 0
                                )
                                if delta is not None
                                else None,
                            }
                        )
        if row["kind"] in {"alias", "new_event"}:
            for reader in MODELS:
                for arm in (
                    "whole_document",
                    "declared_shared",
                    "declared_local",
                    "automatic_shared",
                    "automatic_local",
                ):
                    x, y = (
                        lookup[reader, arm, row["base_id"]],
                        lookup[reader, arm, row["material_id"]],
                    )
                    delta = (
                        y["prediction"] - x["prediction"]
                        if x["prediction"] is not None and y["prediction"] is not None
                        else None
                    )
                    true_base = next(
                        r for r in bank if r["material_id"] == row["base_id"]
                    )
                    expected = (
                        R.true_profile(row)["score"]
                        - R.true_profile(true_base)["score"]
                    )
                    alias_rows.append(
                        {
                            "reader": reader,
                            "target": row["target"],
                            "style": row["style"],
                            "kind": row["kind"],
                            "arm": arm,
                            "material_id": row["material_id"],
                            "predicted_profile_delta": delta,
                            "expected_profile_delta": expected,
                            "before_observed": x["observed_events"],
                            "after_observed": y["observed_events"],
                        }
                    )
    rng = np.random.default_rng(4108)
    for reader, target, style in sorted(
        {
            (p["reader"], p["target"], p["style"])
            for p in predictions
            if p["kind"] == "mixture"
        }
    ):
        rows = [
            r
            for r in bank
            if r["kind"] == "mixture" and r["target"] == target and r["style"] == style
        ]
        for comparator in ("whole_document", "declared_shared", "automatic_shared"):
            paired = [
                (
                    r,
                    lookup[reader, "automatic_local", r["material_id"]],
                    lookup[reader, comparator, r["material_id"]],
                )
                for r in rows
            ]
            paired = [
                (r, a, b)
                for r, a, b in paired
                if a["complete"]
                and b["complete"]
                and a["unit"] is not None
                and b["unit"] is not None
            ]
            if not paired:
                continue
            gains = np.array(
                [
                    (b["unit"] - a["expected_unit"]) ** 2
                    - (a["unit"] - a["expected_unit"]) ** 2
                    for _, a, b in paired
                ]
            )
            f = np.array([r["family"] for r, _, _ in paired])
            unique = np.unique(f)
            estimates = []
            for _ in range(1000):
                chosen = rng.choice(unique, len(unique))
                ix = np.concatenate([np.flatnonzero(f == v) for v in chosen])
                estimates.append(float(gains[ix].mean()))
            comparisons.append(
                {
                    "reader": reader,
                    "target": target,
                    "style": style,
                    "comparator": comparator,
                    "matched_n": len(paired),
                    "automatic_local_mse_improvement": float(gains.mean()),
                    "fixed_program_family_ci": np.quantile(
                        estimates, [0.025, 0.975]
                    ).tolist(),
                }
            )
    write_csv(out / "03/metrics.csv", metrics)
    write_csv(out / "03/same_word_role_contrasts.csv", contrasts)
    write_csv(out / "03/alias_and_new_event.csv", alias_rows)
    write_csv(out / "03/matched_comparisons.csv", comparisons)
    write_json(
        out / "03/summary.json",
        {
            "metrics": metrics,
            "matched_comparisons": comparisons,
            "scope": "new controlled event combinations, not personality validity",
            "ambiguous_scope_excluded_from_unique_gold": sum(
                r["ambiguous"] for r in bank
            ),
        },
    )


def finish(out, source):
    """Verify source immutability, hash finite outputs and leave the goal open."""
    manifest = json.loads((source / "04/artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        if sha(source / name) != digest:
            raise ValueError(f"frozen source changed during HM-04: {name}")
    write_json(
        out / "execution_progress.json",
        {
            "packet": "HM-04",
            "status": "finite_packet_verified",
            "goal_complete": False,
            "old_scope_closed": False,
            "new_fits": 0,
            "next": "inspect role projections and joint binding; then shared target qualification",
        },
    )
    paths = [
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    ]
    write_json(
        out / "04/artifact_hashes.json",
        {str(p.relative_to(out)): sha(p) for p in paths},
    )
    print("HM-04 finite packet finished; no fits or private reads", flush=True)


def main():
    """Execute the finite scope once; hash-checked caches allow safe resumption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    source, originals, bank = prepare(args.out)
    parses = parse_materials(args.out, bank)
    docs, events = observation_inputs(args.out, source, bank, parses)
    for reader in MODELS:
        encode(args.out, reader, args.device)
    predictions = evaluate(args.out, source, originals, bank, docs, events)
    finite_metrics(args.out, bank, predictions)
    finish(args.out, source)


if __name__ == "__main__":
    main()

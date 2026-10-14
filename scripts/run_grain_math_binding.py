#!/usr/bin/env python3
"""Run HM-03 typed action/object experiments with frozen offline readers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

sys.path.insert(0, str(ROOT))
from grain_research.construction import math_binding as B
from grain_research.construction import math_local as L
from grain_research.source_event_graph import stanza_words

DEFAULT = ROOT / "results/grain_math_response_v1/binding_r1"
SPEC = ROOT / "configs/grain_math_binding_v1.json"


def read_bank(path):
    """Read only this public, authored material packet."""
    return [json.loads(s) for s in path.read_text().splitlines()]


def prepare(out):
    """Freeze combinations, scope and input/code hashes before any encoding."""
    spec = json.loads(SPEC.read_text())
    bank = B.material_bank(spec)
    controls = B.relation_controls(spec, bank)
    code = [Path(__file__), ROOT / "grain_research/construction/math_binding.py", SPEC]
    from fetch_grain_syntax_reference import MODELS as SYNTAX

    registry = ROOT / "results/grain_math_response_v1/local_r1/00/reader_registry.json"
    signature = {str(p.relative_to(ROOT)): sha(p) for p in code}
    signature["prior_reader_registry"] = sha(registry)
    signature["parser_inventory"] = sha(SYNTAX / "grain_inventory.json")
    dest = out / "00"
    contract = dest / "protocol_before_run.json"
    if contract.exists():
        old = json.loads(contract.read_text())
        if old["signature"] != signature:
            raise ValueError(
                "binding protocol changed: preserve r1 and use a new output"
            )
        for name, expected in old["material_hashes"].items():
            if sha(dest / name) != expected:
                raise ValueError("changed public materials")
        return bank, controls
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in (("singles.jsonl", bank), ("relations.jsonl", controls)):
        (dest / name).write_text("".join(json.dumps(r) + "\n" for r in data))
    write_json(dest / "reader_registry.json", json.loads(registry.read_text()))
    for p in code:
        target = dest / "code" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write_json(
        contract,
        {
            "signature": signature,
            "material_hashes": {
                name: sha(dest / name) for name in ("singles.jsonl", "relations.jsonl")
            },
            "scope": "HM-03",
            "single_texts": len(bank),
            "relation_texts": len(controls),
            "targets": list(spec),
            "source_validation_unseen_pair_per_target": [32, 16, 16],
            "unseen_object_pairs_per_target": 32,
            "variants_per_pair": 8,
            "kernel_alphas": [0.01, 0.1, 1.0],
            "extra_type_supervision_alpha": 0.01,
            "qualification_midpoint": 0,
            "test_selection": False,
            "result_guided_new_design": True,
            "target": "reported actual action, not personality or verified real-world behavior",
            "parser": "existing frozen Stanza; no target-verb lookup at extraction",
            "supervision": "same relevance labels for whole/additive/tensor/object/action; factor product extra action/object labels separate",
            "api_calls": 0,
            "new_neural_training": 0,
            "private_reads": 0,
        },
    )
    write_json(dest / "exact_binding_witness.json", B.exact_binding_witness())
    print(
        f"protocol frozen: {len(bank)} single and {len(controls)} relation texts",
        flush=True,
    )
    return bank, controls


def parse_public(out, bank, controls):
    """Parse every public case, cache actual words and preserve extraction failures."""
    from fetch_grain_syntax_reference import MODELS as SYNTAX
    from fetch_grain_syntax_reference import PACKAGES, PROCESSORS
    from run_grain_source_event_graph import verify_models

    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    if stanza.__version__ != "1.11.0":
        raise ValueError("unexpected frozen parser version")
    torch.set_num_threads(2)
    cache = out / "01/parses"
    cache.mkdir(parents=True, exist_ok=True)
    pipeline = None
    results = []
    start = time.monotonic()
    calls = 0
    for i, row in enumerate(bank + controls):
        path = cache / f"{i:04d}.json"
        text = row["text"]
        digest = __import__("hashlib").sha256(text.encode()).hexdigest()
        if path.exists():
            result = json.loads(path.read_text())
            if result["text_sha256"] != digest:
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
            words = stanza_words(pipeline(text), text)
            result = {
                "text_sha256": digest,
                "words": words,
                "events": B.dependency_slots(words, text),
            }
            write_json(path, result)
            calls += 1
        results.append(result)
        if (i + 1) % 128 == 0:
            print(
                f"parser {i + 1}/{len(bank) + len(controls)} elapsed={time.monotonic() - start:.1f}s",
                flush=True,
            )
    alignment = []
    for row, parsed in zip(bank, results[: len(bank)], strict=True):
        events = parsed["events"]
        alignment.append(
            {
                "material_id": row["material_id"],
                "target": row["target"],
                "role": row["role"],
                "event_count": len(events),
                "action_exact": len(events) == 1
                and events[0]["action"] == row["action"],
                "object_exact": len(events) == 1
                and events[0]["object"] == row["object"],
            }
        )
    write_csv(out / "01/parser_alignment.csv", alignment)
    write_json(
        out / "01/parser_accounting.json",
        {
            "material_texts": len(results),
            "new_calls_this_invocation": calls,
            "cache_rows": len(results) - calls,
            "elapsed": time.monotonic() - start,
            "syntax_inventory_sha256": sha(SYNTAX / "grain_inventory.json"),
        },
    )
    return results


def encoding_inputs(out, bank, controls, parses):
    """Deduplicate slot strings; public full texts retain original words and roles."""
    rows = [dict(r) for r in bank + controls]
    lookup = {r["text"]: i for i, r in enumerate(rows)}

    def index(slot, value):
        text = f"{slot}: {value}."
        if text not in lookup:
            lookup[text] = len(rows)
            rows.append({"text": text, "kind": "slot", "slot": slot})
        return lookup[text]

    declared = [
        [index("Action", r["action"]), index("Object", r["object"])] for r in bank
    ]
    automatic = [
        [
            [index("Action", e["action"]), index("Object", e["object"])]
            for e in p["events"]
        ]
        for p in parses
    ]
    path = out / "00/material_bank.jsonl"
    content = "".join(json.dumps(r) + "\n" for r in rows)
    if path.exists() and path.read_text() != content:
        raise ValueError("encoding bank changed")
    path.write_text(content)
    write_json(
        out / "01/slot_indices.json",
        {
            "declared": declared,
            "automatic": automatic,
            "encoder_inputs_per_reader": len(rows),
            "single_count": len(bank),
            "relation_count": len(controls),
        },
    )
    return np.array(declared), automatic


def save_model(path, groups):
    """Save only numeric/string arrays, avoiding pickle and executable bundles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"{g}__{k}": v for g, model in groups.items() for k, v in model.items()},
    )


def load_model(path):
    """Read frozen cold programs without evaluation labels."""
    groups = {}
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            group, key = name.split("__", 1)
            groups.setdefault(group, {})[key] = data[name]
    return groups


def fit_target(x, rows, declared, automatic):
    """Fit all programs on source and choose finite alphas on validation only."""
    role = np.array([r["role"] for r in rows])
    bits = np.array([r["bits"] for r in rows])
    sx, vx = role == "source", role == "validation"
    metadata, groups = {}, {}
    for name, y in (
        ("whole_factors", bits.astype(float)),
        ("whole_direct", L.ideal_contribution(bits)),
    ):
        candidates = []
        for alpha in (0.01, 0.1, 1.0):
            model = L.affine_fit(x[sx], y[sx], alpha)
            loss = float(np.mean((L.affine_score(model, x[vx]) - y[vx]) ** 2))
            candidates.append((loss, alpha, model))
        loss, alpha, groups[name] = min(candidates, key=lambda r: r[0])
        metadata[name] = {"alpha": alpha, "validation_mse": loss}
    unique = np.all(bits[:, 1:] == 1, axis=1)
    a, b = declared[0][unique], declared[1][unique]
    pr, py = role[unique], bits[unique, 0].astype(float)
    for name, kind in (
        ("slot_additive", "additive"),
        ("slot_tensor", "tensor"),
        ("object_only", "object"),
        ("action_only", "action"),
    ):
        groups[name], metadata[name] = B.select_kernel(a, b, py, pr, kind)
    src = pr == "source"
    shuffle = py[src][np.random.default_rng(1008).permutation(src.sum())]
    groups["permutation"] = B.fit_kernel(
        a[src], b[src], shuffle, "tensor", float(groups["slot_tensor"]["alpha"])
    )
    ur = [r for r, keep in zip(rows, unique, strict=True) if keep]
    for name, which in (("action_type", 0), ("object_type", 1)):
        key = name.replace("_type", "_index")
        indices = list(
            {r[key]: i for i, r in enumerate(ur) if r["role"] == "source"}.values()
        )
        vectors = a if which == 0 else b
        label = np.array([r[name] for r in ur], float)
        groups[name] = L.affine_fit(vectors[indices], label[indices, None], 0.01)
        metadata[name] = {
            "extra_type_supervision": True,
            "alpha": 0.01,
            "source_unique_slots": len(indices),
        }
    valid = np.array([r is not None for r in automatic])
    pu = unique & valid
    aa = (
        np.array([r[0] for r in automatic if r is not None])
        if valid.any()
        else np.empty((0, x.shape[1]))
    )
    bb = (
        np.array([r[1] for r in automatic if r is not None])
        if valid.any()
        else np.empty((0, x.shape[1]))
    )
    if pu[sx].any() and pu[vx].any():
        order = {i: j for j, i in enumerate(np.flatnonzero(valid))}
        selected = [order[i] for i in np.flatnonzero(pu)]
        for name, kind in (("auto_additive", "additive"), ("auto_tensor", "tensor")):
            groups[name], metadata[name] = B.select_kernel(
                aa[selected], bb[selected], bits[pu, 0].astype(float), role[pu], kind
            )
    metadata["source_rows"] = int(sx.sum())
    metadata["validation_rows"] = int(vx.sum())
    metadata["automatic_valid_rows"] = int(valid.sum())
    return groups, metadata


def fit_all(out, bank, declared_indices, automatic_indices):
    """Freeze program coefficients before reading any held metric."""
    if (out / "02/model_freeze.json").exists():
        for name, expected in json.loads((out / "02/model_freeze.json").read_text())[
            "models"
        ].items():
            if sha(out / "02" / name) != expected:
                raise ValueError("frozen model mismatch")
        return
    metadata = {}
    for reader in MODELS:
        x = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in sorted({r["target"] for r in bank}):
            indices = np.array([i for i, r in enumerate(bank) if r["target"] == target])
            rows = [bank[i] for i in indices]
            di = declared_indices[indices]
            auto = [
                (x[automatic_indices[i][0][0]], x[automatic_indices[i][0][1]])
                if len(automatic_indices[i]) == 1
                else None
                for i in indices
            ]
            groups, meta = fit_target(
                x[indices], rows, (x[di[:, 0]], x[di[:, 1]]), auto
            )
            save_model(out / f"02/{reader}_{target}.npz", groups)
            metadata[f"{reader}_{target}"] = meta
    write_json(out / "02/source_fits.json", metadata)
    paths = sorted((out / "02").glob("*.npz"))
    write_json(
        out / "02/model_freeze.json",
        {
            "models": {p.name: sha(p) for p in paths if not p.name.startswith("._")},
            "selection": "source fit, validation finite alpha; no application metric seen",
            "fitted": list(metadata),
        },
    )
    print("source models frozen, beginning cold application", flush=True)


def relevance_readouts(groups, a, b):
    """Evaluate declared/automatic typed vectors without material truth."""
    result = {
        name: B.score_kernel(groups[name], a, b)
        for name in (
            "slot_additive",
            "slot_tensor",
            "object_only",
            "action_only",
            "permutation",
        )
    }
    va = L.affine_score(groups["action_type"], a)[:, 0]
    vo = L.affine_score(groups["object_type"], b)[:, 0]
    result["extra_type_product"] = (
        2 * np.clip((va + 1) / 2, 0, 1) * np.clip((vo + 1) / 2, 0, 1) - 1
    )
    return result


def crossed_ci(rows, gains):
    """Fixed-program two-way action/object bootstrap, not population uncertainty."""
    a = np.array([r["action_index"] for r in rows])
    o = np.array([r["object_index"] for r in rows])
    ua, uo = np.unique(a), np.unique(o)
    rng, estimates = np.random.default_rng(3108), []
    for _ in range(1000):
        draw_a, draw_o = rng.choice(ua, len(ua)), rng.choice(uo, len(uo))
        wa = {v: int(np.sum(draw_a == v)) for v in ua}
        wo = {v: int(np.sum(draw_o == v)) for v in uo}
        weights = np.array([wa[ai] * wo[oi] for ai, oi in zip(a, o, strict=True)])
        if weights.sum():
            estimates.append(np.average(gains, weights=weights))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def cold_score(out, bank, controls, declared, automatic):
    """Score both holdouts and same-word relation controls with frozen programs."""
    metrics, pairs, predictions, comparisons, relations, parser_metrics = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    max_error, checked = 0.0, 0
    for reader in MODELS:
        x = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in sorted({r["target"] for r in bank}):
            indices = np.array([i for i, r in enumerate(bank) if r["target"] == target])
            rows = [bank[i] for i in indices]
            roles, bits = (
                np.array([r["role"] for r in rows]),
                np.array([r["bits"] for r in rows]),
            )
            di = declared[indices]
            a, b = x[di[:, 0]], x[di[:, 1]]
            groups = load_model(out / f"02/{reader}_{target}.npz")
            z = L.affine_score(groups["whole_factors"], x[indices])
            k = relevance_readouts(groups, a, b)
            scores = {
                "whole_product": B.product_scores(z)[0],
                "whole_direct": L.affine_score(groups["whole_direct"], x[indices])[
                    :, 0
                ],
            }
            factor_views = {"whole_product": z, "whole_direct": z}
            for arm, relevance in k.items():
                scores[arm], factor_views[arm] = B.product_scores(z, relevance)
            auto_valid = np.array([len(automatic[i]) == 1 for i in indices])
            valid_ids = indices[auto_valid]
            aa = np.array([x[automatic[i][0][0]] for i in valid_ids])
            bb = np.array([x[automatic[i][0][1]] for i in valid_ids])
            auto_scores = {}
            for arm in ("auto_additive", "auto_tensor"):
                if arm in groups:
                    ak = B.score_kernel(groups[arm], aa, bb)
                    auto_scores[arm] = B.product_scores(z[auto_valid], ak)
            for domain in ("unseen_pair", "unseen_object"):
                mask = roles == domain
                truth = L.ideal_contribution(bits[mask])[:, 0]
                for arm, sc in scores.items():
                    metrics.append(
                        {
                            "reader": reader,
                            "target": target,
                            "domain": domain,
                            "arm": arm,
                            **B.binding_metrics(
                                sc[mask], factor_views[arm][mask], bits[mask]
                            ),
                        }
                    )
                    for row, value, factors in zip(
                        np.array(rows, object)[mask],
                        sc[mask],
                        factor_views[arm][mask],
                        strict=True,
                    ):
                        predictions.append(
                            {
                                "reader": reader,
                                "target": target,
                                "domain": domain,
                                "arm": arm,
                                "material_id": row["material_id"],
                                "action_index": row["action_index"],
                                "object_index": row["object_index"],
                                "polarity": row["bits"][1],
                                "relevance": row["bits"][0],
                                "prediction": float(value),
                                "relevance_readout": float(factors[0]),
                                "qualified": bool(L.qualify(factors[None])[0]),
                            }
                        )
                for comparator in ("whole_product", "slot_additive", "object_only"):
                    gains = (scores[comparator][mask] - truth) ** 2 - (
                        scores["slot_tensor"][mask] - truth
                    ) ** 2
                    comparisons.append(
                        {
                            "reader": reader,
                            "target": target,
                            "domain": domain,
                            "comparator": comparator,
                            "tensor_mse_improvement": float(gains.mean()),
                            "crossed_fixed_model_ci": crossed_ci(
                                np.array(rows, object)[mask], gains
                            ),
                        }
                    )
                am = roles[auto_valid] == domain
                for arm, (sc, az) in auto_scores.items():
                    parser_metrics.append(
                        {
                            "reader": reader,
                            "target": target,
                            "domain": domain,
                            "arm": arm,
                            "eligible_total": int(mask.sum()),
                            "coverage": float(am.sum() / mask.sum()),
                            **B.binding_metrics(sc[am], az[am], bits[auto_valid][am]),
                        }
                    )
                for arm in ("whole_product", "slot_additive", "slot_tensor"):
                    paired = mask & auto_valid
                    parser_metrics.append(
                        {
                            "reader": reader,
                            "target": target,
                            "domain": domain,
                            "arm": arm + "_parser_matched",
                            "eligible_total": int(mask.sum()),
                            "coverage": float(paired.sum() / mask.sum()),
                            **B.binding_metrics(
                                scores[arm][paired],
                                factor_views[arm][paired],
                                bits[paired],
                            ),
                        }
                    )
                base = mask & np.all(bits[:, 1:] == 1, axis=1)
                chosen = np.flatnonzero(base)
                opposite_obj = np.roll(chosen, 1)
                wrong = B.score_kernel(
                    groups["slot_tensor"], a[chosen], b[opposite_obj]
                )
                for j, ci in enumerate(chosen):
                    pairs.append(
                        {
                            "reader": reader,
                            "target": target,
                            "domain": domain,
                            "material_id": rows[ci]["material_id"],
                            "correct_object_k": float(k["slot_tensor"][ci]),
                            "wrong_object_k": float(wrong[j]),
                            "wrong_object_material": rows[opposite_obj[j]][
                                "material_id"
                            ],
                            "correct_truth_k": int(bits[ci, 0]),
                            "wrong_pair_truth_k": 1
                            if rows[ci]["action_type"]
                            == rows[opposite_obj[j]]["object_type"]
                            == 1
                            else -1,
                        }
                    )
            # Alternate contraction and centering, without calling pair_kernel/score_kernel.
            for arm in (
                "slot_tensor",
                "slot_additive",
                "object_only",
                "action_only",
                "permutation",
            ):
                model = groups[arm]
                ad, bd = (
                    np.einsum("nd,md->nm", a, model["a"]),
                    np.einsum("nd,md->nm", b, model["b"]),
                )
                kind = str(model["kind"])
                matrix = (
                    1 + ad + bd + ad * bd
                    if kind == "tensor"
                    else 1 + ad + bd
                    if kind == "additive"
                    else 1 + bd
                    if kind == "object"
                    else 1 + ad
                )
                recalc = (
                    matrix @ model["dual"]
                    - matrix.mean(axis=1) * model["dual"].sum()
                    - model["column_mean"] @ model["dual"]
                    + model["grand_mean"] * model["dual"].sum()
                    + model["bias"]
                )
                max_error = max(max_error, float(np.max(abs(recalc - k[arm]))))
                checked += len(recalc)
            for j, row in enumerate(controls):
                if row["target"] != target:
                    continue
                whole = L.affine_score(
                    groups["whole_factors"], x[len(bank) + j : len(bank) + j + 1]
                )[0]
                for arm in (
                    "slot_additive",
                    "slot_tensor",
                    "object_only",
                    "extra_type_product",
                ):
                    # Material-declared event identity is a separate observer from automatic parsing.
                    ei = row["event_indices"]
                    ea, eb = x[declared[ei, 0]], x[declared[ei, 1]]
                    ek = relevance_readouts(groups, ea, eb)[arm]
                    ez = np.tile(whole, (len(ei), 1))
                    es, ez = B.product_scores(ez, ek)
                    profile = L.local_profile(
                        [{"event_id": str(v)} for v in ei], es, L.qualify(ez)
                    )
                    relations.append(
                        {
                            "reader": reader,
                            "target": target,
                            "family": row["family"],
                            "swapped": row["swapped"],
                            "expected_present": row["expected_present"],
                            "arm": arm,
                            "route": "declared",
                            "events": len(ei),
                            "predicted_present": profile["observed_events"] > 0,
                            "profile_score": profile["score"],
                            "observed_events": profile["observed_events"],
                            "whole_relevance": float(whole[0]),
                        }
                    )
                ai = automatic[len(bank) + j]
                if ai and "auto_tensor" in groups:
                    ek = B.score_kernel(
                        groups["auto_tensor"],
                        x[[e[0] for e in ai]],
                        x[[e[1] for e in ai]],
                    )
                    es, ez = B.product_scores(np.tile(whole, (len(ai), 1)), ek)
                    profile = L.local_profile(
                        [{"event_id": str(t)} for t in range(len(ai))],
                        es,
                        L.qualify(ez),
                    )
                    relations.append(
                        {
                            "reader": reader,
                            "target": target,
                            "family": row["family"],
                            "swapped": row["swapped"],
                            "expected_present": row["expected_present"],
                            "arm": "auto_tensor",
                            "route": "automatic",
                            "events": len(ai),
                            "predicted_present": profile["observed_events"] > 0,
                            "profile_score": profile["score"],
                            "observed_events": profile["observed_events"],
                            "whole_relevance": float(whole[0]),
                        }
                    )
            # Explicit scalar event-product replay.
            for arm, zz in factor_views.items():
                if arm == "whole_direct":
                    continue
                replay = (
                    zz[:, 1]
                    * np.clip((1 + zz[:, 0]) / 2, 0, 1)
                    * np.clip((1 + zz[:, 2]) / 2, 0, 1)
                    * np.clip((1 + zz[:, 3]) / 2, 0, 1)
                )
                max_error = max(max_error, float(np.max(abs(replay - scores[arm]))))
                checked += len(replay)
    if max_error > 1e-10:
        raise AssertionError(max_error)
    for name, data in (
        ("metrics.csv", metrics),
        ("predictions.csv", predictions),
        ("object_mispairing.csv", pairs),
        ("comparisons.csv", comparisons),
        ("relation_controls.csv", relations),
        ("parser_matched_metrics.csv", parser_metrics),
    ):
        write_csv(out / "03" / name, data)
    write_json(
        out / "03/summary.json",
        {
            "metrics": metrics,
            "comparisons": comparisons,
            "parser_metrics": parser_metrics,
            "scope": "finite authored action units, not human validity",
        },
    )
    write_json(
        out / "04/verification.json",
        {
            "status": "PASS",
            "checked_predictions": checked,
            "max_alternate_formula_error": max_error,
            "scope": "same-agent alternate contractions and products, not independent human review",
        },
    )
    paths = sorted(
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    )
    write_json(
        out / "04/artifact_hashes.json",
        {str(p.relative_to(out)): sha(p) for p in paths},
    )
    print(
        f"cold binding scores verified: {checked} predictions, error={max_error:.3g}",
        flush=True,
    )


def main():
    """Run a bounded public packet; retain every negative result and old protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    bank, controls = prepare(args.out)
    parses = parse_public(args.out, bank, controls)
    declared, automatic = encoding_inputs(args.out, bank, controls, parses)
    for reader in MODELS:
        encode(args.out, reader, args.device)
    fit_all(args.out, bank, declared, automatic)
    cold_score(args.out, bank, controls, declared, automatic)
    write_json(
        args.out / "execution_progress.json",
        {
            "packet": "HM-03",
            "status": "finite_packet_verified",
            "goal_complete": False,
            "old_scope_closed": False,
            "next": "inspect object and relation contrasts, select next scientific mechanism",
        },
    )


if __name__ == "__main__":
    main()

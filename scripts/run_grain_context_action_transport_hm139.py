#!/usr/bin/env python3
"""Source freeze -> fresh encoder -> cold numeric scores -> native mechanism test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import anonymous_bundle as AB
from grain_research.construction import math_context_action_transport as M
from scripts import run_grain_context_program_fibers_hm138 as PRIOR
from scripts import run_grain_same_evidence_hm137 as H

SPEC = ROOT / "configs/grain_context_action_transport_hm139.json"
CODE = (
    "configs/grain_context_action_transport_hm139.json",
    "docs/GRAIN_CONTEXT_ACTION_TRANSPORT_PLAN_HM139.md",
    "grain_research/construction/math_context_action_transport.py",
    "scripts/run_grain_context_action_transport_hm139.py",
    "tests/test_grain_context_action_transport.py",
)


def contrasts(events):
    """Integrate all24 separate four-corner author contrasts in fixed order."""
    x = np.asarray(events, float).reshape(24, 2, 2, -1)
    return np.stack([M.interaction(c) for c in x])


def check(out):
    """Verify sealed code and scientific source/material dependencies."""
    manifest = H.read(out / "manifest.json")
    for p, digest in (manifest["code"] | manifest["inputs"]).items():
        if H.sha(p) != digest:
            raise ValueError("Changed HM139 input: " + p)
    return manifest


def prepare(out):
    """Freeze fresh materials and source functions before any application read."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if out.exists():
        raise FileExistsError("Preserve previous/failed run")
    spec = H.read(SPEC)
    src = ROOT / spec["source"]
    PRIOR.fixed_inputs(H.read(PRIOR.SPEC))
    parent, _, g, native, directions = H.arrays(src)
    previous = ROOT / parent["spec"]["previous"]
    qmanifest = H.read(previous / "00/manifest.json")
    query = H.read(previous / "00/queries.json")
    sources = H.read(previous / "00/sources.json")
    base = sources[0]["rows"][0]
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    old_texts = {r["text"] for r in H.read(src / "00/records.json")}
    sham_source = M.balanced_labels(24, spec["source_sham_seed"])
    sham_app = M.balanced_labels(24, spec["application_sham_seed"])
    rows, records = [], []
    for corpus, bank, family, b, c in itertools.product(
        range(2), range(2), range(6), (-1, 1), (-1, 1)
    ):
        definition = spec["bank_definitions"][bank]
        bi, ci = (b + 1) // 2, (c + 1) // 2
        if corpus == 0:
            text = (
                definition["prefix"]
                + " "
                + definition["actions"][family][bi]
                + ".\n"
                + definition["cause_notes"][ci]
            )
        else:
            text = (
                definition["prefix"]
                + " freely "
                + definition["actual_actions"][family]
                + ". Unused alternative: "
                + definition["options"][family][bi]
                + ".\n"
                + definition["availability_notes"][ci]
            )
        if text in old_texts or any(r["text"] == text for r in records):
            raise ValueError("New public bodies must be distinct and disjoint")
        tokens = tok.encode(text, add_special_tokens=False)
        if len(tokens) > len(base["history"]):
            raise ValueError("New text exceeds full history support; no truncation")
        ids = list(base["input_ids"])
        for i in base["history"]:
            ids[i] = tok.pad_token_id
        for i, token in zip(base["history"][: len(tokens)], tokens, strict=True):
            ids[i] = token
        stratum = (corpus * 2 + bank) * 6 + family
        records.append(
            {
                "id": len(records),
                "corpus": corpus,
                "bank": bank,
                "family": family,
                "b": b,
                "c": c,
                "sham": int(sham_app[stratum]),
                "text": text,
                "tokens": len(tokens),
            }
        )
        rows.append(
            {
                "id": len(rows),
                "history": base["history"],
                "current": base["current"],
                "padding": base["history"][len(tokens) :],
                "input_ids": ids,
            }
        )
    gflat = g.reshape(96, 15)
    yflat = native[:, :, :, :, :, 1, :, :, :2].reshape(96, 28)
    real = np.repeat(np.repeat([0, 1], 12), 4)
    dictionary = {
        "global": M.features(gflat),
        "real": M.features(gflat, real),
        "sham": M.features(gflat, np.repeat(sham_source, 4)),
    }
    y = contrasts(yflat)
    models = {
        name: M.fit_actions(contrasts(x), y, spec["rtol"])
        for name, x in dictionary.items()
    }
    out.mkdir(parents=True)
    (out / "native").mkdir()
    H.write(out / "records.json", records)
    H.write(out / "rows.json", rows)
    H.write(out / "queries.json", query)
    H.write(
        out / "source_bundle.json",
        {
            "models": models,
            "targets": parent["previous_targets"] + ["action_fact"],
            "units": ["score", "logodds"],
            "actors": ["A", "B"],
            "sham_source": sham_source.tolist(),
            "sham_application": sham_app.tolist(),
            "condition_rule": "supplied experimental regime;not inferred",
            "intercept_status": "fixed_zero_not_identified_by_contrasts",
        },
    )
    np.savez(
        out / "source_actions.npz",
        y=y,
        **{name: contrasts(x) for name, x in dictionary.items()},
        directions=directions,
    )
    input_paths = [
        src / "02/seal.json",
        src / "verification.json",
        previous / "00/manifest.json",
        previous / "00/queries.json",
        previous / "00/sources.json",
        *[
            out / n
            for n in (
                "records.json",
                "rows.json",
                "queries.json",
                "source_bundle.json",
                "source_actions.npz",
            )
        ],
    ]
    H.write(
        out / "manifest.json",
        {
            "spec": spec,
            "encoder": parent["spec"],
            "reader_model": qmanifest["model"],
            "reader_depth": qmanifest["model_blocks"],
            "code": {str(ROOT / p): H.sha(ROOT / p) for p in CODE},
            "inputs": {str(p): H.sha(p) for p in input_paths},
            "native_dependency_code": {
                str(ROOT / p): H.sha(ROOT / p) for p in qmanifest["code"]
            },
            "source_fits": 3,
            "source_native_channels": 28,
            "new_neural_training": 0,
            "original_scope_complete": False,
        },
    )
    H.write(
        out / "status.json",
        {
            "state": "PREPARED",
            "reader_batches": 0,
            "reader_seconds": 0,
            "encoder_seconds": 0,
            "pending": None,
        },
    )
    print(
        "HM139 source functions frozen",
        {name: m["rank"] for name, m in models.items()},
        "new texts",
        len(records),
        flush=True,
    )


def encode(out):
    """Collect new E5 only; no native reader outputs exist in this stage."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    manifest, state = check(out), H.read(out / "status.json")
    if state["state"] != "PREPARED":
        raise ValueError("No repeated application encoding")
    spec, enc = manifest["spec"], manifest["encoder"]
    start = time.monotonic()
    state.update(state="ENCODING", pending="load_E5")
    H.write(out / "status.json", state)
    try:
        for p, digest in H.read(ROOT / spec["source"] / "00/manifest.json")[
            "model_files"
        ].items():
            if H.sha(p) != digest:
                raise ValueError("Frozen encoder cache changed")
        torch.set_num_threads(4)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(
            enc["encoder_snapshot"], local_files_only=True
        )
        model = (
            AutoModel.from_pretrained(
                enc["encoder_snapshot"],
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        values, receipts = [], []
        records = H.read(out / "records.json")
        for first in range(0, len(records), 8):
            if time.monotonic() - start > spec["max_encoder_seconds"]:
                raise TimeoutError("Original encoder cap")
            batch = tok(
                [enc["encoder_prefix"] + r["text"] for r in records[first : first + 8]],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if batch["input_ids"].shape[1] > enc["maximum_length"]:
                raise ValueError("No silent token truncation")
            state["pending"] = first
            H.write(out / "status.json", state)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                raw = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy().astype(float)
            values.append(raw)
            receipts.append(
                {
                    "first": first,
                    "count": len(raw),
                    "token_width": int(batch["input_ids"].shape[1]),
                }
            )
        raw = np.concatenate(values)
        with np.load(out / "source_actions.npz", allow_pickle=False) as z:
            directions = z["directions"]
        g = AB.normalize(raw) @ directions
        np.savez(out / "encoder.npz", raw=raw, g=g)
        cold = [
            {"g": g[i].tolist(), "real_context": r["corpus"], "sham_context": r["sham"]}
            for i, r in enumerate(records)
        ]
        H.write(out / "cold_input.json", cold)
        H.write(out / "encoder_receipts.json", receipts)
        state.update(
            state="ENCODED", pending=None, encoder_seconds=time.monotonic() - start
        )
        H.write(out / "status.json", state)
        print("ENCODED", len(g), "texts", state["encoder_seconds"], flush=True)
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    except BaseException as error:
        state.update(
            state="FAILED_NOT_RETRIED",
            encoder_seconds=time.monotonic() - start,
            error=repr(error),
        )
        H.write(out / "status.json", state)
        raise


def score(out):
    """Cold calculation uses only coordinates/contexts; preflight hashes all inputs."""
    check(out)
    state = H.read(out / "status.json")
    if state["state"] != "ENCODED" or list((out / "native").iterdir()):
        raise ValueError("Predictions must be sealed before any new native read")
    predictions = M.cold_score(
        H.read(out / "source_bundle.json"), H.read(out / "cold_input.json")
    )
    np.savez(out / "cold_scores.npz", **predictions)
    files = [
        out / n
        for n in (
            "cold_input.json",
            "encoder.npz",
            "encoder_receipts.json",
            "cold_scores.npz",
        )
    ]
    H.write(out / "cold_seal.json", {str(p): H.sha(p) for p in files})
    state.update(state="COLD_SCORED")
    H.write(out / "status.json", state)
    print("COLD_SCORED before native labels;4 fixed arms", flush=True)


def collect(out):
    """Full native observation after source and cold predictions are immutable."""
    from grain_research.vector_mechanisms.navigation import MODEL
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, state = check(out), H.read(out / "status.json")
    if state["state"] != "COLD_SCORED" or state["reader_batches"]:
        raise ValueError("No uncertain/native replay")
    for p, digest in (
        H.read(out / "cold_seal.json") | manifest["native_dependency_code"]
    ).items():
        if H.sha(p) != digest:
            raise ValueError("Cold predictions/dependency changed")
    H.PREV.H.verify_hashes(MODEL, manifest["reader_model"])
    rows, query = H.read(out / "rows.json"), H.read(out / "queries.json")
    try:
        with collector_guard(out):
            started = time.monotonic()
            observer = load_observer()
            if len(observer.model.model.layers) != manifest["reader_depth"]:
                raise ValueError("Original reader depth required")
            for r in rows:
                if time.monotonic() - started > manifest["spec"]["max_reader_seconds"]:
                    raise TimeoutError("Original cumulative reader cap")
                state.update(
                    state="READING",
                    pending=r["id"],
                    reader_seconds=time.monotonic() - started,
                )
                H.write(out / "status.json", state)
                reads, primitive = H.PREV.A.native_batch(
                    observer, r, query["suffixes"], query["alphabet"], 7, 17
                )
                logits = np.array([v["candidate_logits"] for v in reads])
                norm = np.array([v["full_vocab_log_normalizer"] for v in reads])
                path = out / "native" / f"r{r['id']:03d}.npz"
                np.savez(
                    path,
                    logits=logits,
                    normalizer=norm,
                    readings=H.PREV.A.units(logits, norm),
                    **{
                        n: primitive[n]
                        for n in (
                            "blocks",
                            "endpoints",
                            "support_hash",
                            "execution_length",
                        )
                    },
                )
                H.write(
                    path.with_suffix(".json"),
                    {"id": r["id"], "endpoints": len(reads), "sha256": H.sha(path)},
                )
                state.update(
                    reader_batches=state["reader_batches"] + 1,
                    pending=None,
                    reader_seconds=time.monotonic() - started,
                )
                H.write(out / "status.json", state)
                if state["reader_batches"] % 12 == 0:
                    print(
                        "HM139 reader",
                        state["reader_batches"],
                        "/96",
                        round(state["reader_seconds"], 2),
                        flush=True,
                    )
        state.update(state="COLLECTED", pending=None)
        H.write(out / "status.json", state)
        files = list((out / "native").iterdir())
        H.write(out / "native_seal.json", {str(p): H.sha(p) for p in files})
    except BaseException as error:
        state.update(state="FAILED_NOT_RETRIED", error=repr(error))
        H.write(out / "status.json", state)
        raise


def observations(out):
    """Authenticate all actual primitive native receipts, recompute readouts."""
    depth = check(out)["reader_depth"]
    state = H.read(out / "status.json")
    if (
        state["state"] != "COLLECTED"
        or state["reader_batches"] != 96
        or state["pending"] is not None
    ):
        raise ValueError("Complete terminal96 required")
    for p, digest in H.read(out / "native_seal.json").items():
        if H.sha(p) != digest:
            raise ValueError("Changed native primitive")
    y, error = [], 0.0
    rows, query = H.read(out / "rows.json"), H.read(out / "queries.json")
    for i in range(96):
        p = out / "native" / f"r{i:03d}.npz"
        receipt = H.read(p.with_suffix(".json"))
        if (
            receipt["id"] != i
            or receipt["endpoints"] != 14
            or receipt["sha256"] != H.sha(p)
        ):
            raise ValueError("Native address receipt mismatch")
        with np.load(p, allow_pickle=False) as z:
            masks = [
                H.PREV.A.permissions(rows[i], len(s), int(z["execution_length"]), 7)
                for s in query["suffixes"]
            ]
            hashes = np.stack(
                [
                    np.frombuffer(
                        hashlib.sha256(mask.tobytes()).digest(), dtype=np.uint8
                    )
                    for mask in masks
                ]
            )
            endpoints = [
                len(rows[i]["input_ids"]) + len(s) - 1 for s in query["suffixes"]
            ]
            if (
                not np.array_equal(z["support_hash"], hashes)
                or not np.array_equal(z["endpoints"], endpoints)
                or not np.array_equal(z["blocks"], np.arange(depth))
            ):
                raise ValueError("Actual suffix permissions/address/depth changed")
            v = H.PREV.A.units(z["logits"], z["normalizer"])
            error = max(error, float(np.max(abs(v - z["readings"]))))
            y.append(M.person_channels(v)[..., :2].reshape(28))
    return np.stack(y), error


def derive(out, verify=False):
    """Compare frozen action laws and independently integrate every positive law."""
    manifest = check(out)
    for p, digest in H.read(out / "cold_seal.json").items():
        if H.sha(p) != digest:
            raise ValueError("Changed before-outcome prediction")
    y, native_error = observations(out)
    observed = contrasts(y)
    with np.load(out / "cold_scores.npz", allow_pickle=False) as z:
        predictions = {name: z[name].copy() for name in z.files}
    bundle = H.read(out / "source_bundle.json")
    fresh = M.cold_score(bundle, H.read(out / "cold_input.json"))
    cold_error = max(float(np.max(abs(predictions[n] - fresh[n]))) for n in fresh)
    plus, minus = M.positive_laws()
    tables, actions, integration_error = [], [], 0.0
    corpus = np.repeat([0, 1], 12)
    bank = np.tile(np.repeat([0, 1], 6), 2)
    channel = list(itertools.product(("A", "B"), bundle["targets"], bundle["units"]))
    slices = [
        ("all", np.arange(24)),
        ("cause", np.where(corpus == 0)[0]),
        ("opportunity", np.where(corpus == 1)[0]),
        ("new_bank0", np.where(bank == 0)[0]),
        ("new_bank1", np.where(bank == 1)[0]),
    ]
    for arm, values in predictions.items():
        delta = contrasts(values)
        for s in range(24):
            for j, (actor, target, unit) in enumerate(channel):
                actual = sum(
                    float((plus - minus)[b, c] * values[4 * s + 2 * b + c, j])
                    for b, c in np.ndindex(2, 2)
                )
                integration_error = max(integration_error, abs(actual - delta[s, j]))
                actions.append(
                    {
                        "arm": arm,
                        "stratum": s,
                        "corpus": int(corpus[s]),
                        "bank": int(bank[s]),
                        "family": s % 6,
                        "actor": actor,
                        "target": target,
                        "unit": unit,
                        "observed": float(observed[s, j]),
                        "predicted": float(delta[s, j]),
                        "signed_error": float(delta[s, j] - observed[s, j]),
                    }
                )
        for name, ix in slices:
            for j, (actor, target, unit) in enumerate(channel):
                tables.append(
                    {
                        "arm": arm,
                        "slice": name,
                        "actor": actor,
                        "target": target,
                        "unit": unit,
                        **M.compare(delta[ix, j], observed[ix, j]),
                    }
                )
    # Compare action construction to fresh coordinate contrasts, separately from native truth.
    cold_records = H.read(out / "cold_input.json")
    g = np.array([r["g"] for r in cold_records])
    real = np.array([r["real_context"] for r in cold_records])
    sham = np.array([r["sham_context"] for r in cold_records])
    ff = {
        "global": M.features(g),
        "real": M.features(g, real),
        "sham": M.features(g, sham),
        "wrong_real": M.features(g, 1 - real),
    }
    operator_error = max(
        float(
            np.max(
                abs(
                    contrasts(ff[n])
                    @ bundle["models"]["real" if n == "wrong_real" else n]["beta"]
                    - contrasts(predictions[n])
                )
            )
        )
        for n in ff
    )
    with np.load(out / "source_actions.npz", allow_pickle=False) as z:
        source_errors = {}
        for name in ("global", "real", "sham"):
            x = z[name]
            scale = np.array(bundle["models"][name]["scale"])
            beta_alt = (
                np.linalg.lstsq(x / scale, z["y"], rcond=manifest["spec"]["rtol"])[0]
                / scale[:, None]
            )
            source_errors[name] = float(
                np.max(abs(x @ beta_alt - x @ bundle["models"][name]["beta"]))
            )
    summary = {
        "status": "VERIFIED_FINITE_ACTION_TRANSPORT_NOT_PERSONALITY",
        "source_ranks": {n: m["rank"] for n, m in bundle["models"].items()},
        "source_action_error_rms": {
            n: m["source_error_rms"] for n, m in bundle["models"].items()
        },
        "events": 96,
        "native_endpoints": 1344,
        "new_source_matrix_fits": 3,
        "source_fit_units": "native full-program actions;not psychological labels",
        "cold_before_native": True,
        "metric_rows": len(tables),
        "action_rows": len(actions),
        "native_readout_error": native_error,
        "cold_replay_error": cold_error,
        "operator_action_error": operator_error,
        "positive_law_error": integration_error,
        "alternate_source_prediction_error": source_errors,
        "reader_seconds": H.read(out / "status.json")["reader_seconds"],
        "encoder_seconds": H.read(out / "status.json")["encoder_seconds"],
        "original_scope_complete": False,
    }
    maximum = max(
        native_error,
        cold_error,
        operator_error,
        integration_error,
        *source_errors.values(),
    )
    if maximum > manifest["spec"]["tolerance"]:
        raise ValueError("Numerical verification failed")
    if verify:
        for p, digest in H.read(out / "output_seal.json").items():
            if H.sha(p) != digest:
                raise ValueError("Changed scientific output")
        if summary != H.read(out / "summary.json"):
            raise ValueError("Summary replay differs")
        import csv

        for filename, expected in (("metrics.csv", tables), ("actions.csv", actions)):
            with (out / filename).open() as f:
                actual = list(csv.DictReader(f))
            if len(actual) != len(expected) or any(
                any(
                    row[k] != ("" if value is None else str(value))
                    for k, value in e.items()
                )
                for row, e in zip(actual, expected, strict=True)
            ):
                raise ValueError("Full result table replay differs")
        print("READONLY VERIFIED HM139", summary, flush=True)
    else:
        H.table(out / "metrics.csv", tables)
        H.table(out / "actions.csv", actions)
        H.write(out / "summary.json", summary)
        files = [out / n for n in ("metrics.csv", "actions.csv", "summary.json")]
        H.write(out / "output_seal.json", {str(p): H.sha(p) for p in files})
        print("DERIVED HM139", summary, flush=True)


def main():
    """Expose finite stages; no automatic recovery of an uncertain read."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "encode", "score", "collect", "derive", "verify")
    )
    args = parser.parse_args()
    out = ROOT / H.read(SPEC)["out"]
    with threadpool_limits(limits=1):
        if args.stage == "verify":
            derive(out, verify=True)
        else:
            globals()[args.stage](out)


if __name__ == "__main__":
    main()

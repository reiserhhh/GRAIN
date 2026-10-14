#!/usr/bin/env python3
"""HM46: acquire new crossed-construct materials and actual vector interventions."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_construct_routing as M
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from grain_research.vector_mechanisms.navigation import write_json as atomic_json
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_construct_routing_v1.json"
OUT = ROOT / "results/grain_math_response_v1/construct_routing_r1"
CODE = (
    "configs/grain_math_construct_routing_v1.json",
    "docs/GRAIN_MULTI_CONSTRUCT_ROUTING_PLAN_V1.md",
    "grain_research/construction/math_construct_routing.py",
    "scripts/run_grain_math_construct_routing.py",
    "scripts/verify_grain_math_construct_routing.py",
    "tests/test_grain_math_construct_routing.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/navigation.py",
    "scripts/run_grain_math_span_binding.py",
)


def write(path, value):
    """Atomically save a recoverable JSON unit without overwriting historical packets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def save(path, meta, arrays):
    """Commit arrays before metadata; metadata hash binds each completed call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    npz = path.with_suffix(".npz")
    tmp = npz.with_suffix(".npz.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(npz)
    write(path, {**meta, "array_sha256": digest(npz)})


def prepare(out):
    """Freeze complete new materials and matched-token patches before any model call."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Keep existing packets; do not overwrite preparation")
    spec = read_json(SPEC)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = {"labels": list("0123456789"), "values": (np.arange(10) / 9).tolist()}
    tokens = [tokenizer.encode(x, add_special_tokens=False) for x in alphabet["labels"]]
    if any(len(t) != 1 for t in tokens) or len({t[0] for t in tokens}) != 10:
        raise ValueError("Distinct single-digit candidates required")
    alphabet["token_ids"] = [t[0] for t in tokens]
    rows = []
    for doc in M.documents(spec):
        previous = None
        for target in spec["targets"]:
            rendered, enc, offset = M.render(doc["body"], target, tokenizer)
            start, stop = doc["property_interval"]
            positions = M.token_positions(
                enc["offset_mapping"], offset + start, offset + stop
            )
            body_positions = M.token_positions(
                enc["offset_mapping"], offset, offset + len(doc["body"])
            )
            endpoint = body_positions[-1]
            question_start = M.token_positions(
                enc["offset_mapping"], rendered.index("\nQuestion: ") + 1, len(rendered)
            )[0]
            prefix = enc["input_ids"][:question_start]
            if previous is not None and prefix != previous:
                raise ValueError("Future query altered the actual body token prefix")
            previous = prefix
            rows.append(
                {
                    **doc,
                    "unit": len(rows),
                    "query": target,
                    "input_ids": list(enc["input_ids"]),
                    "rendered": rendered,
                    "property_positions": positions,
                    "body_endpoint": endpoint,
                    "question_start": question_start,
                    "property_tokens": [enc["input_ids"][p] for p in positions],
                }
            )
    pairs = M.patch_pairs(rows)
    if len(rows) != 1152:
        raise ValueError("Full192texts crossed with six queries required")
    for directory in ("00/code", "01/units", "02/patches", "03"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    write(out / "00/design.json", rows)
    write(out / "00/patch_pairs.json", pairs)
    code = {}
    for name in CODE:
        p = out / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(p)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM46",
            "spec": spec,
            "alphabet": alphabet,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code": code,
            "design_sha256": digest(out / "00/design.json"),
            "patch_design_sha256": digest(out / "00/patch_pairs.json"),
            "model_config_sha256": digest(MODEL / "config.json"),
            "new_texts": 192,
            "source_texts": 96,
            "application_texts": 96,
            "source_forwards": 576,
            "application_forwards": 576,
            "patch_forwards": 144,
            "noop_forwards": 6,
            "no_training_or_api": True,
            "human_personality_truth": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "PREPARED:192 new texts /1152 query crossings /150 intervention controls",
        flush=True,
    )


def check(out):
    """Check only this packet's frozen inputs, not an unrelated ancestry backlog."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(out / "00/code" / name) != sha or digest(ROOT / name) != sha:
            raise ValueError("Changed frozen code " + name)
    for file, key in (
        ("design.json", "design_sha256"),
        ("patch_pairs.json", "patch_design_sha256"),
    ):
        if digest(out / "00" / file) != m[key]:
            raise ValueError("Changed prepared material")
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Changed local reader config")
    return m


def observe(observer, row, alphabet, layer, patch=None):
    """Run a frozen full reader, optionally replacing one declared block16 carrier."""
    import torch

    device = next(observer.model.parameters()).device
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    captured = {}

    def hook(_module, _args, output):
        h = hidden_tensor(output)
        if patch is not None:
            positions, donor = patch
            if np.shape(donor) != (len(positions), h.shape[-1]):
                raise ValueError("Exact donor carrier shape required")
            h = h.clone()
            h[0, positions] = torch.tensor(donor, dtype=h.dtype, device=h.device)
        captured.update(
            property=h[0, row["property_positions"]].detach().cpu().numpy().copy(),
            endpoint=h[0, row["body_endpoint"]].detach().cpu().numpy().copy(),
            query_end=h[0, -1].detach().cpu().numpy().copy(),
        )
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = observer.model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = observer.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        observer.calls += 1
        torch.mps.synchronize()
        native = native_readout(output.logits[0, -1].float().cpu().numpy(), alphabet)
        return native, captured
    finally:
        handle.remove()


def collect(out):
    """Complete source then application then finite patches; uncertain calls are not retried."""
    m = check(out)
    status = read_json(out / "run_status.json")
    if status["in_flight"] is not None:
        raise RuntimeError(
            "Uncertain in-flight call needs inspection; do not duplicate it"
        )
    if status["status"] == "COLLECTED":
        return
    started, previous = time.monotonic(), status["model_seconds"]
    observer = load_observer()
    write(out / "01/observer.json", observer.manifest())
    rows = read_json(out / "00/design.json")
    lookup = {r["unit"]: r for r in rows}

    def call(key, row, patch=None):
        if (
            status["forward_calls"] >= m["spec"]["max_forwards"]
            or previous + time.monotonic() - started >= m["spec"]["max_model_seconds"]
        ):
            raise TimeoutError("Finite packet budget reached; preserve checkpoint")
        status.update(
            in_flight=key,
            status="COLLECTING",
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        native, arrays = observe(
            observer, row, m["alphabet"], m["spec"]["layer"], patch
        )
        status["forward_calls"] += 1
        return native, arrays

    def commit(path, meta, arrays):
        save(path, meta, arrays)
        status.update(
            in_flight=None, model_seconds=previous + time.monotonic() - started
        )
        write(out / "run_status.json", status)

    def load(unit):
        p = out / f"01/units/unit_{unit:04d}.json"
        meta = read_json(p)
        if digest(p.with_suffix(".npz")) != meta["array_sha256"]:
            raise ValueError("Changed capture")
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
            return meta, {k: a[k].copy() for k in a.files}

    for role in ("source", "application"):
        if role == "application":
            barrier = read_json(out / "01/source_barrier.json")
            for file, sha in barrier["files"].items():
                if digest(out / file) != sha:
                    raise ValueError("Changed source before application")
        for r in rows:
            if r["role"] != role:
                continue
            path = out / f"01/units/unit_{r['unit']:04d}.json"
            if path.exists():
                load(r["unit"])
                continue
            native, arrays = call(f"native/{r['unit']}", r)
            commit(path, {"unit": r["unit"], "role": role, "native": native}, arrays)
            if status["forward_calls"] % 96 == 0:
                print(f"{role}:{status['forward_calls']} actual forwards", flush=True)
        if role == "source" and not (out / "01/source_barrier.json").exists():
            files = {
                str(p.relative_to(out)): digest(p)
                for p in (out / "01/units").glob("unit_*")
                if not p.name.startswith("._")
            }
            if len(files) != 1152:
                raise ValueError("Source must include576 complete metadata/array units")
            write(
                out / "01/source_barrier.json",
                {
                    "files": files,
                    "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )

    for pair in read_json(out / "00/patch_pairs.json"):
        r, d = lookup[pair["receiver_unit"]], lookup[pair["donor_unit"]]
        rm, ra = load(r["unit"])
        dm, da = load(d["unit"])
        for arm in ("property", "endpoint", "random_property"):
            p = out / f"02/patches/pair_{pair['pair_id']:03d}_{arm}.json"
            if p.exists():
                continue
            if arm == "endpoint":
                pos, donor = [r["body_endpoint"]], da["endpoint"][None]
                norm = float(np.linalg.norm(donor - ra["endpoint"][None]))
            else:
                pos, donor = r["property_positions"], da["property"]
                delta = donor - ra["property"]
                norm = float(np.linalg.norm(delta))
                if arm == "random_property":
                    donor = ra["property"] + M.norm_matched(
                        delta, m["spec"]["random_seed"] + pair["pair_id"]
                    )
            native, arrays = call(f"patch/{pair['pair_id']}/{arm}", r, (pos, donor))
            commit(
                p,
                {
                    **pair,
                    "arm": arm,
                    "native": native,
                    "receiver_native": rm["native"],
                    "donor_native": dm["native"],
                    "patch_norm": norm,
                    "actual_patch_norm": float(
                        np.linalg.norm(
                            donor
                            - (
                                ra["endpoint"][None]
                                if arm == "endpoint"
                                else ra["property"]
                            )
                        )
                    ),
                },
                arrays,
            )
    for target in m["spec"]["targets"]:
        r = next(
            r
            for r in rows
            if r["role"] == "application"
            and r["query"] == r["content"] == target
            and r["wording"] == r["polarity"] == r["other"] == r["imagined"] == 0
        )
        p = out / f"02/patches/noop_{target}.json"
        if not p.exists():
            rm, ra = load(r["unit"])
            native, arrays = call(
                "noop/" + target, r, (r["property_positions"], ra["property"])
            )
            commit(
                p,
                {
                    "unit": r["unit"],
                    "query": target,
                    "native": native,
                    "receiver_native": rm["native"],
                },
                arrays,
            )
    status.update(
        status="COLLECTED", model_seconds=previous + time.monotonic() - started
    )
    write(out / "run_status.json", status)
    print(json.dumps(status), flush=True)


def analyze(out):
    """Measure all conditional response relations without selecting a predictor."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError(
            "Require the full finite acquisition, not partial file presence"
        )
    rows = read_json(out / "00/design.json")
    targets = m["spec"]["targets"]
    scores = np.full((2, 2, 6, 6, 2, 2, 2), np.nan)
    native_rows, prefix_errors = [], []
    bodies = {}
    for r in rows:
        p = out / f"01/units/unit_{r['unit']:04d}.json"
        meta = read_json(p)
        n = meta["native"]
        i = (
            r["family"],
            r["wording"],
            targets.index(r["content"]),
            targets.index(r["query"]),
            r["polarity"],
            r["other"],
            r["imagined"],
        )
        scores[i] = n["conditional_score"]
        native_rows.append(
            {
                k: r[k]
                for k in (
                    "unit",
                    "doc_id",
                    "role",
                    "content",
                    "query",
                    "wording",
                    "polarity",
                    "other",
                    "imagined",
                )
            }
            | n
        )
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
            body = np.concatenate((a["property"].ravel(), a["endpoint"].ravel()))
        if r["doc_id"] in bodies:
            prefix_errors.append(float(np.max(abs(body - bodies[r["doc_id"]]))))
        else:
            bodies[r["doc_id"]] = body
    coefficients, matrix_rows, matrices, spectra = [], [], {}, []
    for f, w, c, q in itertools.product(range(2), range(2), range(6), range(6)):
        beta, arithmetic = M.split_cube(scores[f, w, c, q])
        for term, value in zip(M.TERMS, beta, strict=True):
            coefficients.append(
                {
                    "role": "source" if f == 0 else "application",
                    "wording": w,
                    "content": targets[c],
                    "query": targets[q],
                    "term": term,
                    "value": float(value),
                    **arithmetic,
                }
            )
    for f, w, o, a in itertools.product(range(2), repeat=4):
        matrix = scores[f, w, :, :, 0, o, a] - scores[f, w, :, :, 1, o, a]
        key = f"{f}_{w}_{o}_{a}"
        matrices[key] = matrix
        spectra.append(
            {
                "role": "source" if f == 0 else "application",
                "wording": w,
                "other": o,
                "imagined": a,
                **M.spectrum(matrix),
            }
        )
        for c, q in itertools.product(range(6), repeat=2):
            matrix_rows.append(
                {
                    "role": "source" if f == 0 else "application",
                    "wording": w,
                    "other": o,
                    "imagined": a,
                    "content": targets[c],
                    "query": targets[q],
                    "polarity_delta": float(matrix[c, q]),
                }
            )
    agreement = []
    for o, a in itertools.product(range(2), repeat=2):
        src = np.mean([matrices[f"0_{w}_{o}_{a}"] for w in range(2)], axis=0)
        app = np.mean([matrices[f"1_{w}_{o}_{a}"] for w in range(2)], axis=0)
        denom = float(np.linalg.norm(src) * np.linalg.norm(app))
        us, _, vs = np.linalg.svd(src, full_matrices=False)
        ua, _, va = np.linalg.svd(app, full_matrices=False)
        overlap = float((us[:, 0] @ app @ vs[0]) ** 2)
        app_energy = float(np.sum(app * app))
        agreement.append(
            {
                "other": o,
                "imagined": a,
                "source_application_frobenius_cosine": float(np.sum(src * app) / denom)
                if denom
                else None,
                "source_mode_application_squared_share": overlap / app_energy
                if app_energy
                else None,
                "left_mode_abs_cosine": float(abs(us[:, 0] @ ua[:, 0])),
                "right_mode_abs_cosine": float(abs(vs[0] @ va[0])),
            }
        )
    patch_rows, noops = [], []
    for p in sorted((out / "02/patches").glob("*.json")):
        if p.name.startswith("._"):
            continue
        r = read_json(p)
        delta = (
            r["native"]["conditional_score"] - r["receiver_native"]["conditional_score"]
        )
        if p.name.startswith("noop"):
            noops.append(delta)
            continue
        d = rows[r["receiver_unit"]]
        patch_rows.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "pair_id",
                        "axis",
                        "arm",
                        "receiver_unit",
                        "donor_unit",
                        "patch_norm",
                        "actual_patch_norm",
                    )
                },
                "content": d["content"],
                "polarity": d["polarity"],
                "receiver_other": d["other"],
                "receiver_imagined": d["imagined"],
                "native_context_delta": r["donor_native"]["conditional_score"]
                - r["receiver_native"]["conditional_score"],
                "patch_delta": delta,
                "candidate_mass": r["native"]["candidate_mass"],
            }
        )
    patch_summary = []
    for axis, arm in itertools.product(
        ("other", "imagined"), ("property", "endpoint", "random_property")
    ):
        selected = [r for r in patch_rows if r["axis"] == axis and r["arm"] == arm]
        x, y = (
            np.array([r["native_context_delta"] for r in selected]),
            np.array([r["patch_delta"] for r in selected]),
        )
        patch_summary.append(
            {
                "axis": axis,
                "arm": arm,
                "n_pairs": len(selected),
                "native_delta_rms": float(np.sqrt(np.mean(x * x))),
                "patch_delta_rms": float(np.sqrt(np.mean(y * y))),
                "native_patch_dot": float(x @ y),
                "through_origin_slope": float(x @ y / (x @ x)) if x @ x else None,
                "sign_agreement": int(np.sum(x * y > 0)),
                "opposite_sign": int(np.sum(x * y < 0)),
            }
        )
    for name, table in (
        ("readouts.csv", native_rows),
        ("cube_coefficients.csv", coefficients),
        ("response_matrices.csv", matrix_rows),
        ("interventions.csv", patch_rows),
        ("intervention_summary.csv", patch_summary),
    ):
        write_csv(out / "03" / name, table)
    write(out / "03/spectra.json", spectra)
    write(out / "03/agreement.json", agreement)
    summary = {
        "status": "FINITE_HM46_ACQUIRED_AND_ANALYZED_NOT_GLOBAL",
        "new_texts": 192,
        "native_forwards": len(native_rows),
        "patches": len(patch_rows),
        "noops": len(noops),
        "body_prefix_max_query_difference": max(prefix_errors),
        "noop_max_score_difference": max(abs(x) for x in noops),
        "minimum_candidate_mass": min(r["candidate_mass"] for r in native_rows),
        "median_candidate_mass": float(
            np.median([r["candidate_mass"] for r in native_rows])
        ),
        "cube_max_reconstruction_error": max(
            r["reconstruction_max_error"] for r in coefficients
        ),
        "cube_max_parseval_error": max(r["parseval_error"] for r in coefficients),
        "model_run_status": read_json(out / "run_status.json"),
        "new_fits": 0,
        "human_personality_accuracy": False,
        "application_patch_donors_are_diagnostic": True,
        "HM45_executed": False,
        "agreement": agreement,
        "patch_summary": patch_summary,
    }
    write(out / "03/summary.json", summary)
    hashes = {
        str(p.relative_to(out)): digest(p)
        for folder in ("00", "01", "02", "03")
        for p in (out / folder).rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    }
    write(out / "03/artifact_hashes.json", hashes)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Run one explicit stage or the complete finite packet; never auto-tune results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        for stage, function in (
            ("prepare", prepare),
            ("collect", collect),
            ("analyze", analyze),
        ):
            if args.stage in (stage, "all"):
                function(args.out)


if __name__ == "__main__":
    main()

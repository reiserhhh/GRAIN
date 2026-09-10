#!/usr/bin/env python3
"""NEXT32: bounded source Jacobians and frozen portable baseline candidates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.comment_carrier import partition, score
from grain_research.vector_mechanisms.complete_pair import (
    context_kernel,
    fit_context,
    ridge,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.joint_readout import centered_logits
from grain_research.vector_mechanisms.query_baseline import (
    candidate_jacobian,
    contract,
    predict_affine,
    source_affine,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_comment_carrier import FIELD
from scripts.run_grain_complete_pair import contexts
from scripts.run_grain_conditional_energy import PARENT, source_inputs
from scripts.run_grain_configuration_parity import anchor_coordinates
from scripts.run_grain_joint_carrier_readout import LEVEL, OLD, STATES, check, verify
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

PLAN = ROOT / "docs/GRAIN_QUERY_DIRECTED_BASELINE_PLAN_V1.md"
HELPER = ROOT / "grain_research/vector_mechanisms/query_baseline.py"
EPS = (0.001, 0.002)


def prepare(out):
    """Freeze membership, source-only input layout and implementation before calls."""
    verify(PARENT)
    out.mkdir(exist_ok=False)
    (out / "private_packets").mkdir()
    pairs, _, _ = source_inputs()
    designs = {r["index"]: r for r in read(OLD / "private_design.json")}
    authors = [r | {"role": designs[r["index"]]["role"]} for r in pairs[::2]]
    if [r["role"] for r in authors].count("reference") != 4 or len(authors) != 24:
        raise ValueError("Four old reference and20 evaluation N authors required")
    blocks, signature = contexts(authors)
    np.savez_compressed(out / "private_context.npz", **blocks)
    write(out / "private_design.json", authors)
    files = [
        PLAN,
        HELPER,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/complete_pair.py",
        ROOT / "grain_research/vector_mechanisms/context_carrier.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
        ROOT / "scripts/run_grain_mean_null_path.py",
        ROOT / "scripts/run_grain_configuration_parity.py",
        ROOT / "scripts/run_grain_complete_pair.py",
    ]
    old = read(OLD / "manifest.json")
    states = {
        str((STATES / f"author_{r['index']:03d}.npz").relative_to(ROOT)): sha(
            STATES / f"author_{r['index']:03d}.npz"
        )
        for r in authors
    }
    for p, digest in states.items():
        if old["sources"][p] != digest:
            raise ValueError("Source-state lineage changed")
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_DIRECTED_BASELINE_V1",
            "plan_commit": "e174ae8",
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "state_hashes": states,
            "reference_sha256": sha(LEVEL / "reference.npz"),
            "old_endpoint_freeze_sha256": sha(OLD / "endpoint_freeze.json"),
            "model_sha256": old["model_sha256"],
            "prepared_files": {
                p: sha(out / p) for p in ("private_context.npz", "private_design.json")
            },
            "roles": {"source": 4, "evaluation": 20},
            "context_signature": signature,
            "max_forwards": 24,
            "max_backwards": 40,
            "max_seconds": 180,
            "finite_difference": {
                "steps": EPS,
                "method": "central",
                "direction": "next source displacement, unit token-coordinate RMS",
                "block": "even source slots M, odd R",
            },
            "human_labels_read": False,
            "prefix_calls": 0,
            "new_neural_training": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forwards": 0,
            "backwards": 0,
            "seconds": 0,
            "complete_sources": 0,
        },
    )
    print({"status": "PREPARED", "source_authors": 4, "evaluation_authors": 20})


def verified(out):
    """Validate frozen code/input lineage before source measurement or construction."""
    m = read(out / "manifest.json")
    for p, d in m["code_hashes"].items():
        check(ROOT / p, d)
    for p, d in m["prepared_files"].items():
        check(out / p, d)
    check(LEVEL / "reference.npz", m["reference_sha256"])
    check(OLD / "endpoint_freeze.json", m["old_endpoint_freeze_sha256"])
    return m


def collect(out):
    """Collect four derivative packets with bounded, live-accounted neural work."""
    m = verified(out)
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print({"status": "ALREADY_COMPLETE", "new_calls": 0})
        return
    if prior["seconds"] >= m["max_seconds"] or prior["forwards"] >= m["max_forwards"]:
        raise ValueError("Cumulative source budget exhausted; do not restart the queue")
    check(MODEL / "model.safetensors", m["model_sha256"])
    authors = read(out / "private_design.json")
    sources = [r for r in authors if r["role"] == "reference"]
    designs = {r["index"]: r for r in read(OLD / "private_design.json")}
    with np.load(out / "private_context.npz") as z:
        blocks = dict(z)
    source_indices = [j for j, r in enumerate(authors) if r["role"] == "reference"]
    with np.load(LEVEL / "reference.npz") as z:
        refs = dict(z)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, old = observer.manifest(), read(FIELD / "observer.json")
    for k in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[k] != old[k]:
            raise ValueError("Frozen observer execution changed")
    write(out / "observer.json", now)
    start = time.monotonic()
    backwards = prior["backwards"]
    complete = 0

    def status(value):
        payload = {
            "status": value,
            "forwards": prior["forwards"] + observer.calls,
            "backwards": backwards,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "complete_sources": complete,
            "prefix_calls": 0,
            "human_labels_read": False,
        }
        write(out / "run_status.json", payload)
        print(payload, flush=True)

    def guard():
        if (
            prior["seconds"] + time.monotonic() - start >= m["max_seconds"]
            or prior["forwards"] + observer.calls >= m["max_forwards"]
        ):
            raise TimeoutError("Source measurement resource boundary")

    def backward_done():
        nonlocal backwards
        backwards += 1
        if backwards > m["max_backwards"]:
            raise TimeoutError("Backward budget reached")
        guard()

    try:
        for slot, r in enumerate(sources):
            dst = out / "private_packets" / f"source_{slot:02d}.npz"
            if dst.exists():
                check(dst, read(dst.with_suffix(".json"))["arrays_sha256"])
                complete += 1
                continue
            guard()
            statepath = STATES / f"author_{r['index']:03d}.npz"
            check(statepath, m["state_hashes"][str(statepath.relative_to(ROOT))])
            v = designs[r["index"]]["views"][0]
            g, _, _, rest = partition(v["groups"], v["p_mask"])
            with np.load(statepath) as z:
                h = torch.tensor(z["view_0"][None], device="mps")
            native = anchor_coordinates(
                h,
                np.array(v["p_mask"], bool),
                refs["neuroticism__Z"],
                refs["neuroticism__h"],
            )
            base = (
                native
                + mean_null_direction(
                    native[0], [x.tolist() for x in g], list(range(12))
                )[None]
            )
            kwargs = cached_suffix_kwargs(observer.model, base)
            logits, jm, jr = candidate_jacobian(
                observer,
                base,
                kwargs,
                v["alphabet"],
                g,
                rest,
                on_backward=backward_done,
            )
            oldscore = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")[
                "scores"
            ]["zero"]
            replay = float(
                np.max(
                    abs(
                        centered_logits(logits)
                        - centered_logits(oldscore["candidate_logits"])
                    )
                )
            )
            score_replay = abs(
                score(logits, v["alphabet"]["values"]) - oldscore["conditional_score"]
            )
            if replay > 2e-5 or score_replay > 2e-6:
                raise ValueError("Native baseline replay failed")
            ix, nxt = source_indices[slot], source_indices[(slot + 1) % 4]
            region = "M" if slot % 2 == 0 else "R"
            d = (blocks[region][nxt] - blocks[region][ix]).reshape(-1, 1536)
            direction = torch.zeros_like(base)
            if region == "M":
                weight = np.array([len(x) for x in g])
                scale = np.sqrt(np.sum(d * d * weight[:, None]) / (weight.sum() * 1536))
                small = torch.tensor(d / scale, dtype=base.dtype, device="mps")
                for positions, value in zip(g, small, strict=True):
                    direction[:, positions] = value
                analytic = contract(jm, small.cpu().numpy())
            else:
                small = torch.tensor(
                    d / np.sqrt(np.mean(d * d)), dtype=base.dtype, device="mps"
                )
                direction[:, rest] = small
                analytic = contract(jr, small.cpu().numpy())
            fd = []
            for eps in EPS:
                guard()
                plus = replay_suffix(
                    observer, base + eps * direction, kwargs, v["alphabet"]
                )
                guard()
                minus = replay_suffix(
                    observer, base - eps * direction, kwargs, v["alphabet"]
                )
                fd.append(
                    (
                        centered_logits(plus["candidate_logits"])
                        - centered_logits(minus["candidate_logits"])
                    )
                    / (2 * eps)
                )
            np.savez_compressed(
                dst,
                jm=jm,
                jr=jr,
                logits=centered_logits(logits),
                fd=np.array(fd),
                analytic=analytic,
            )
            write(
                dst.with_suffix(".json"),
                {
                    "source_slot": slot,
                    "index": r["index"],
                    "author_id": r["author_id"],
                    "region": region,
                    "logit_replay_max": replay,
                    "score_replay_max": score_replay,
                    "arrays_sha256": sha(dst),
                },
            )
            complete += 1
            status("RUNNING")
            del h, native, base, direction
            torch.mps.empty_cache()
        status("COMPLETE")
        files = sorted(
            p
            for p in (out / "private_packets").iterdir()
            if p.name.startswith("source_") and p.suffix in (".npz", ".json")
        )
        write(
            out / "source_freeze.json",
            {
                "files": {str(p.relative_to(out)): sha(p) for p in files},
                "status_sha256": sha(out / "run_status.json"),
                "manifest_sha256": sha(out / "manifest.json"),
            },
        )
    except TimeoutError:
        status("PAUSED_RESOURCE_BOUNDARY")
    except BaseException:
        status("FAILED")
        raise


def construct(out, name):
    """Freeze a complete affine predictor plus source-matched Ridge/mean controls."""
    verified(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Four complete source Jacobians required")
    f = read(out / "source_freeze.json")
    check(out / "run_status.json", f["status_sha256"])
    for p, d in f["files"].items():
        check(out / p, d)
    dest = out / name
    dest.mkdir(exist_ok=False)
    authors = read(out / "private_design.json")
    train = np.array([r["role"] == "reference" for r in authors])
    with np.load(out / "private_context.npz") as z:
        blocks = dict(z)
    z0, jm, jr = [], [], []
    for slot in range(4):
        with np.load(out / "private_packets" / f"source_{slot:02d}.npz") as z:
            z0.append(z["logits"])
            jm.append(z["jm"])
            jr.append(z["jr"])
    z0 = np.array(z0)
    means = blocks["M"].reshape(24, 12, 1536)
    retained = blocks["R"].reshape(24, 136, 1536)
    model = source_affine(means[train], retained[train], z0, np.array(jm), np.array(jr))
    affine = predict_affine(means[~train], retained[~train], model)
    kernel = fit_context({k: v[train] for k, v in blocks.items()})
    k = context_kernel({k: v[train] for k, v in blocks.items()}, kernel)
    weights, scale = ridge(k, z0 - z0.mean(0), 0.01)
    kernel["weights"] = weights
    kernel["target_mean"] = z0.mean(0)
    kernel["kernel_scale"] = np.array(scale)
    predicted_kernel = (
        z0.mean(0)
        + context_kernel({k: v[~train] for k, v in blocks.items()}, kernel) @ weights
    )
    np.savez_compressed(dest / "private_model.npz", **model)
    np.savez_compressed(dest / "private_kernel_model.npz", **kernel)
    np.savez_compressed(
        dest / "private_inputs.npz", **{k: v[~train] for k, v in blocks.items()}
    )
    np.savez(
        dest / "private_predictions.npz",
        **affine,
        kernel_logits=centered_logits(predicted_kernel),
        mean_logits=np.repeat(z0.mean(0)[None], 20, 0),
    )
    replay = predict_affine(
        means[~train], retained[~train], dict(np.load(dest / "private_model.npz"))
    )
    max_replay = max(float(np.max(abs(affine[k] - replay[k]))) for k in affine)
    if max_replay != 0:
        raise ValueError("Serialized affine replay changed")
    write(
        dest / "construction_summary.json",
        {
            "source_authors": 4,
            "evaluation_authors": 20,
            "numerical_fits": 1,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "coefficient_replay_max": max_replay,
            "affine_assembly_error": float(
                np.max(abs(affine["logits"] - affine["per_source"].mean(1)))
            ),
            "kernel_penalty": 0.01,
        },
    )
    files = sorted(
        p
        for p in dest.iterdir()
        if p.name.startswith("private_") or p.name == "construction_summary.json"
    )
    write(
        dest / "prediction_freeze.json",
        {
            "files": {p.name: sha(p) for p in files},
            "source_freeze_sha256": sha(out / "source_freeze.json"),
            "constructor_sha256": sha(Path(__file__)),
        },
    )
    print(
        {
            "status": "PREDICTIONS_FROZEN",
            "held_authors": 20,
            "serialized_replay_max": max_replay,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "collect", "construct"))
    p.add_argument("--name", default="logic_query_baseline_v1")
    p.add_argument("--construction", default="numerical")
    a = p.parse_args()
    out = BASE / a.name
    if a.stage == "prepare":
        prepare(out)
    elif a.stage == "collect":
        collect(out)
    else:
        construct(out, a.construction)

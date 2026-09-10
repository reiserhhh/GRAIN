#!/usr/bin/env python3
"""NEXT19/20: frozen joint-logit kernels and reference retained-state functionals."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.comment_carrier import (
    partition,
    reference_state,
    score,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.joint_readout import (
    centered_logits,
    fit_joint,
    predict_joint,
)
from grain_research.vector_mechanisms.kernel_action import (
    contrast_features,
    training_geometry,
)
from grain_research.vector_mechanisms.query_functional import (
    matrix_response,
    matrix_variants,
    path_state,
    region_gradient,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_configuration_parity import anchor_coordinates
from scripts.run_grain_mean_null_path import mean_null_direction, simpson
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

OLD = BASE / "logic_comment_carrier_v1_r2"
PROJECTION = BASE / "logic_kernel_action_v1/r2_replay"
STATES = BASE / "source_anchor_diagnosis_v1/private_packets"
PLAN = ROOT / "docs/GRAIN_LOGIC_JOINT_CARRIER_READOUT_PLAN_V1.md"
PAIRS = {"adjacent": (0, 1), "distant": (0, 11)}
ARMS = ("fourier_bag", "fourier_position")


def check(path, digest):
    """Reject changed frozen inputs without printing their private contents."""
    if sha(path) != digest:
        raise ValueError(f"Frozen input changed: {path.name}")


def prepare(out):
    """Snapshot source lineage and current code before numerical construction."""
    old = read(OLD / "manifest.json")
    ef = read(OLD / "endpoint_freeze.json")
    pf = read(PROJECTION / "input_freeze.json")
    if read(OLD / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete measured quartet required")
    inputs = {}

    def add(path, digest=None):
        if digest is not None:
            check(path, digest)
        inputs[str(path.relative_to(ROOT))] = sha(path)

    for name in ("manifest.json", "endpoint_freeze.json", "private_design.json"):
        add(OLD / name)
    check(OLD / "private_design.json", old["design_sha256"])
    for name, digest in ef["packets"].items():
        add(OLD / "private_packets" / name, digest)
    add(PROJECTION / "input_freeze.json")
    add(PROJECTION / "private_targets.csv", pf["target_sha256"])
    add(PROJECTION / "private_directions.npz", pf["directions_sha256"])
    for name, digest in pf["packet_sha256"].items():
        add(PROJECTION / name, digest)
    for row in read(OLD / "private_design.json"):
        p = STATES / f"author_{row['index']:03d}.npz"
        add(p, old["sources"][str(p.relative_to(ROOT))])
    add(LEVEL / "reference.npz", old["reference_sha256"])
    files = [
        Path(__file__),
        PLAN,
        ROOT / "scripts/analyze_grain_joint_carrier_readout.py",
    ]
    files += [
        ROOT / "grain_research/vector_mechanisms" / f"{n}.py"
        for n in (
            "joint_readout",
            "kernel_action",
            "query_functional",
            "comment_carrier",
            "context_carrier",
            "trace",
        )
    ]
    files += [
        ROOT / "scripts" / f"{n}.py"
        for n in (
            "run_grain_configuration_parity",
            "run_grain_mean_null_path",
        )
    ]
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_nodes").mkdir()
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_JOINT_CARRIER_READOUT_V1",
            "inputs": inputs,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "model_sha256": old["model_sha256"],
            "source_outcome_sha256": old["source_outcome_sha256"],
            "human_labels_for_fit_or_selection": False,
            "suffix_limit": 40,
            "seconds_limit": 180,
            "reference_authors": 4,
            "evaluation_authors": 20,
        },
    )
    print({"status": "PREPARED", "old_authors": 48, "new_prefix_calls": 0})


def verify(out):
    """Check immutable protocol and all cached states/endpoints, not source Y."""
    m = read(out / "manifest.json")
    for key in ("inputs", "code_hashes"):
        for name, digest in m[key].items():
            check(ROOT / name, digest)
    return m


def freeze(out, name, files):
    """Record numerical outputs before any questionnaire file is opened."""
    write(
        out / name,
        {"human_outcomes_read": False, "files": {p.name: sha(p) for p in files}},
    )


def joint(out, destination=None):
    """Fit eight author-disjoint tensor maps to measured N logit actions only."""
    verify(out)
    dest = out if destination is None else destination
    if (dest / "joint_freeze.json").exists():
        raise ValueError("Preserve frozen joint predictions")
    dest.mkdir(exist_ok=True)
    started = time.monotonic()
    designs = {r["index"]: r for r in read(OLD / "private_design.json")}
    data = pd.read_csv(PROJECTION / "private_targets.csv", dtype={"author_id": str})
    data = data[data["query"] == "neuroticism"].sort_values(["author_id", "view"])
    if len(data) != 72 or data.groupby("author_id").fold.nunique().ne(1).any():
        raise ValueError("Old24 N authors with three author-disjoint views required")
    packets = {}
    for i in data["index"].unique():
        with np.load(PROJECTION / "private_projected" / f"author_{i:03d}.npz") as z:
            packets[i] = dict(z)
    with np.load(PROJECTION / "private_directions.npz") as z:
        eta = z["neuroticism__eta"].copy()
    rows = data.to_dict("records")
    means = [packets[r["index"]][f"c_{r['view']}"].mean(0) for r in rows]
    variances = [(packets[r["index"]][f"d_{r['view']}"] ** 2).mean(0) for r in rows]
    pairs = []
    for r in [r for r in rows if r["view"] == 0]:
        d = designs[r["index"]]
        p = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")
        if str(p["author_id"]) != r["author_id"] or p["query"] != r["query"]:
            raise ValueError("Endpoint/projection identity mismatch")
        for name, (i, j) in PAIRS.items():
            logits = [
                np.array(p["scores"][k]["candidate_logits"])
                for k in (
                    "zero",
                    f"s{i}",
                    f"s{j}",
                    name,
                )
            ]
            mixed = centered_logits(logits[3] - logits[1] - logits[2] + logits[0])
            pairs.append(
                {
                    **r,
                    "pair": name,
                    "i": i,
                    "j": j,
                    "sizes": [len(g) for g in d["views"][0]["groups"]],
                    "mixed": mixed,
                    "z_add": logits[1] + logits[2] - logits[0],
                    "actual_score": score(logits[3], p["alphabet"]["values"]),
                    "values": p["alphabet"]["values"],
                }
            )
    targets = np.array([r["mixed"] for r in pairs])
    output, models = [], []
    for fold in range(4):
        keep = np.array([r["fold"] != fold for r in pairs])
        if keep.sum() != 36 or (~keep).sum() != 12:
            raise ValueError("18 train/6 evaluation authors required")
        center, scale = training_geometry(
            means, variances, np.array([r["fold"] != fold for r in rows])
        )
        control = np.array(
            [
                targets[keep & np.array([p["pair"] == r["pair"] for p in pairs])].mean(
                    0
                )
                for r in pairs
            ]
        )
        for arm in ARMS:
            left, right = [], []
            for r in pairs:
                z = packets[r["index"]]
                offsets = np.cumsum([0] + r["sizes"])
                if offsets[-1] != len(z["c_0"]):
                    raise ValueError(
                        "Projected body rows do not match comment partition"
                    )
                f = []
                for idx in (r["i"], r["j"]):
                    sl = slice(offsets[idx], offsets[idx + 1])
                    phi = contrast_features(
                        z["c_0"][sl],
                        z["d_0"][sl],
                        z["position_0"][sl],
                        eta,
                        center,
                        scale,
                        arm,
                    )
                    f.append(np.concatenate(phi) * r["sizes"][idx] / offsets[-1])
                left.append(f[0])
                right.append(f[1])
            left, right = np.array(left), np.array(right)
            model = fit_joint(left[keep], right[keep], targets[keep])
            pred = predict_joint(left[~keep], right[~keep], model)
            swap_error = np.max(
                abs(pred - predict_joint(right[~keep], left[~keep], model))
            )
            if (
                swap_error > 1e-12
                or np.max(abs(predict_joint(left[~keep] * 0, right[~keep], model))) != 0
            ):
                raise ValueError("Pair structural identities failed")
            models.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "center": center.tolist(),
                    "geometry_scale": scale.tolist(),
                    "swap_error": float(swap_error),
                    **{
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in model.items()
                    },
                }
            )
            for k, ix in enumerate(np.flatnonzero(~keep)):
                r = pairs[ix]
                choices = {arm: pred[k]}
                if arm == ARMS[0]:
                    choices.update(
                        zero=np.zeros_like(pred[k]), pair_train_mean=control[ix]
                    )
                for label, prediction in choices.items():
                    output.append(
                        {k: r[k] for k in ("author_id", "query", "fold", "pair")}
                        | {
                            "arm": label,
                            "actual_mixed": r["mixed"].tolist(),
                            "predicted_mixed": prediction.tolist(),
                            "z_add": r["z_add"].tolist(),
                            "values": r["values"],
                            "actual_score": r["actual_score"],
                            "oracle_singletons_reconstruction": score(
                                r["z_add"] + prediction, r["values"]
                            ),
                        }
                    )
    if len(output) != 192 or len(models) != 8:
        raise ValueError("Expected48 pairs/four arms/eight fits")
    write(dest / "private_joint_predictions.json", output)
    write(dest / "private_joint_models.json", models)
    freeze(
        dest,
        "joint_freeze.json",
        [dest / "private_joint_predictions.json", dest / "private_joint_models.json"],
    )
    print(
        {
            "status": "JOINT_FROZEN",
            "rows": len(output),
            "fits": len(models),
            "seconds": round(time.monotonic() - started, 2),
            "neural_calls": 0,
        }
    )


def donors_and_design():
    """Load only source states; reference membership is inherited, not chosen by Y."""
    design = [r for r in read(OLD / "private_design.json") if r["query"] == "TF_cont"]
    donors = []
    for r in design:
        if r["role"] == "reference":
            with np.load(STATES / f"author_{r['index']:03d}.npz") as z:
                v = r["views"][0]
                donors.append((z["view_0"].copy(), v["groups"], v["p_mask"]))
    if len(donors) != 4 or len(design) != 24:
        raise ValueError("Frozen4/20 TF source panel required")
    return donors, design


def carrier_collect(out):
    """Measure only four reference gradient paths, bounded to40 suffix pairs."""
    m = verify(out)
    if (out / "carrier_nodes_freeze.json").exists():
        print("Reference nodes already frozen; no new neural calls")
        return
    check(MODEL / "model.safetensors", m["model_sha256"])
    donors, design = donors_and_design()
    with np.load(LEVEL / "reference.npz") as z:
        refs = dict(z)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now = observer.manifest()
    old = read(BASE / "centered_adjoint_field_v1/observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[key] != old[key]:
            raise ValueError("Observer changed")
    write(out / "observer.json", now)
    prior = (
        read(out / "carrier_status.json")
        if (out / "carrier_status.json").exists()
        else {"calls": 0, "seconds": 0}
    )
    start = time.monotonic()
    audits = []
    device = next(observer.model.parameters()).device

    def status(value):
        payload = {
            "status": value,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "prefix_calls": 0,
            "model_training": False,
            "reference_paths": len(audits),
        }
        write(out / "carrier_status.json", payload)
        print(payload, flush=True)

    try:
        for row in [r for r in design if r["role"] == "reference"]:
            v = row["views"][0]
            g, _, body, rest = partition(v["groups"], v["p_mask"])
            with np.load(STATES / f"author_{row['index']:03d}.npz") as z:
                h = torch.tensor(z["view_0"][None], device=device)
            native = anchor_coordinates(
                h, np.array(v["p_mask"], bool), refs["TF_cont__Z"], refs["TF_cont__h"]
            )
            base = (
                native
                + mean_null_direction(
                    native[0], [x.tolist() for x in g], list(range(12))
                )[None]
            )
            ref_np, count = reference_state(
                base[0].cpu().numpy(), g, v["p_mask"], donors
            )
            if count or len(rest) != 136:
                raise ValueError("Exact136 retained positions required")
            ref = torch.tensor(ref_np[None], device=device)
            d = base[0, rest].cpu().numpy().astype(float) - ref_np[rest].astype(float)
            kwargs = cached_suffix_kwargs(observer.model, native)
            mask = np.zeros(native.shape[1], bool)
            mask[rest] = True
            endpoints = read(
                OLD / "private_packets" / f"author_{row['index']:03d}.json"
            )["scores"]
            for body_case in (0, 1):
                context = ref.clone()
                if body_case:
                    context[:, body] = base[:, body]
                expected = [
                    endpoints[k]["conditional_score"]
                    for k in (("b10", "zero") if body_case else ("b00", "b01"))
                ]
                gradients, values = {}, {}

                def node(
                    t,
                    row=row,
                    body_case=body_case,
                    context=context,
                    ref=ref,
                    rest=rest,
                    base=base,
                    mask=mask,
                    kwargs=kwargs,
                    v=v,
                    gradients=gradients,
                    values=values,
                ):
                    """Save each node immediately; bind this path's state on entry."""
                    p = (
                        out
                        / "private_nodes"
                        / f"ref_{row['index']:03d}_m{body_case}_t{int(t * 100):03d}.npz"
                    )
                    if p.exists():
                        with np.load(p) as z:
                            val, grad = float(z["score"]), z["gradient"].copy()
                    else:
                        if (
                            prior["calls"] + observer.calls >= m["suffix_limit"]
                            or prior["seconds"] + time.monotonic() - start
                            >= m["seconds_limit"]
                        ):
                            raise TimeoutError(
                                "Bounded gradient resource limit reached"
                            )
                        val, grad = region_gradient(
                            observer,
                            context,
                            path_state(ref[:, rest], base[:, rest], t),
                            mask,
                            kwargs,
                            v["alphabet"],
                        )
                        np.savez_compressed(p, score=val, gradient=grad, fraction=t)
                        status("RUNNING")
                    gradients[t], values[t] = grad.astype(float), val

                for t in (0.0, 0.5, 1.0):
                    node(t)
                endpoint_error = max(
                    abs(values[0.0] - expected[0]), abs(values[1.0] - expected[1])
                )
                if endpoint_error > 2e-6:
                    raise ValueError("Exact reference endpoint replay failed")
                actual = expected[1] - expected[0]
                tolerance = max(1e-4, 0.02 * abs(actual))
                first_error = abs(matrix_response(simpson(gradients), d) - actual)
                if first_error > tolerance:
                    for t in (0.25, 0.75):
                        node(t)
                matrix = simpson(gradients)
                residual = matrix_response(matrix, d) - actual
                np.savez_compressed(
                    out
                    / "private_nodes"
                    / f"ref_{row['index']:03d}_m{body_case}_integral.npz",
                    matrix=matrix,
                )
                audits.append(
                    {
                        "index": row["index"],
                        "body_case": body_case,
                        "nodes": len(gradients),
                        "actual": actual,
                        "initial_error": first_error,
                        "final_error": abs(residual),
                        "tolerance": tolerance,
                        "passes": abs(residual) <= tolerance,
                        "endpoint_error": endpoint_error,
                    }
                )
        write(out / "private_integration_audit.json", audits)
        status("COMPLETE")
        write(
            out / "carrier_nodes_freeze.json",
            {
                "human_outcomes_read": False,
                "files": {
                    str(p.relative_to(out)): sha(p)
                    for p in sorted((out / "private_nodes").glob("ref_*.npz"))
                },
                "audit_sha256": sha(out / "private_integration_audit.json"),
            },
        )
    except BaseException as exc:
        status(
            "PAUSED_RESOURCE_BOUNDARY"
            if isinstance(exc, TimeoutError)
            else "FAILED_RETAINED"
        )
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def carrier_read(out, destination=None):
    """Freeze all24 numerical readouts; evaluation authors supply states but no gradients."""
    verify(out)
    dest = out if destination is None else destination
    dest.mkdir(exist_ok=True)
    if (dest / "carrier_freeze.json").exists():
        raise ValueError("Preserve frozen carrier predictions")
    frozen = read(out / "carrier_nodes_freeze.json")
    for name, digest in frozen["files"].items():
        check(out / name, digest)
    check(out / "private_integration_audit.json", frozen["audit_sha256"])
    donors, design = donors_and_design()
    matrices = []
    target_mean = []
    for row in [r for r in design if r["role"] == "reference"]:
        pair = []
        for body_case in (0, 1):
            with np.load(
                out
                / "private_nodes"
                / f"ref_{row['index']:03d}_m{body_case}_integral.npz"
            ) as z:
                pair.append(z["matrix"])
        matrices.append(np.mean(pair, axis=0))
        s = read(OLD / "private_packets" / f"author_{row['index']:03d}.json")["scores"]
        target_mean.append(
            (
                s["b01"]["conditional_score"]
                - s["b00"]["conditional_score"]
                + s["zero"]["conditional_score"]
                - s["b10"]["conditional_score"]
            )
            / 2
        )
    matrix = np.mean(matrices, axis=0)
    variants, info = matrix_variants(matrix)
    np.savez_compressed(dest / "private_carrier_matrices.npz", **variants)
    write(dest / "carrier_matrix_summary.json", info)
    output = []
    for row in design:
        v = row["views"][0]
        g, _, _, rest = partition(v["groups"], v["p_mask"])
        with np.load(STATES / f"author_{row['index']:03d}.npz") as z:
            h = z["view_0"].copy()
        ref, count = reference_state(h, g, v["p_mask"], donors)
        if count or len(rest) != 136:
            raise ValueError("Retained role grid changed")
        delta = h[rest].astype(float) - ref[rest].astype(float)
        s = read(OLD / "private_packets" / f"author_{row['index']:03d}.json")["scores"]
        native, b10, b01, b00 = [
            s[k]["conditional_score"] for k in ("zero", "b10", "b01", "b00")
        ]
        actual = (b01 - b00 + native - b10) / 2
        body_allocation = (b10 - b00 + native - b01) / 2
        predictions = {k: matrix_response(m, delta) for k, m in variants.items()}
        predictions.update(zero=0.0, reference_mean=float(np.mean(target_mean)))
        for arm, pred in predictions.items():
            output.append(
                {k: row[k] for k in ("author_id", "query", "role")}
                | {
                    "arm": arm,
                    "actual_retained": actual,
                    "predicted_retained": pred,
                    "actual_baseline": b00,
                    "actual_body_allocated": body_allocation,
                    "actual_complete_b": native,
                    "oracle_baseline_body_reconstruction": b00 + body_allocation + pred,
                }
            )
    pd.DataFrame(output).to_csv(dest / "private_carrier_predictions.csv", index=False)
    files = [
        dest / name
        for name in (
            "private_carrier_matrices.npz",
            "private_carrier_predictions.csv",
            "carrier_matrix_summary.json",
        )
    ]
    freeze(dest, "carrier_freeze.json", files)
    print(
        {
            "status": "CARRIER_READOUT_FROZEN",
            "references": 4,
            "evaluation": 20,
            "rows": len(output),
            "evaluation_gradient_calls": 0,
            **info,
        }
    )


def main():
    """Execute one finite phase; private artifacts remain below ignored results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "phase",
        choices=("prepare", "joint", "carrier-collect", "carrier-read", "replay"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_joint_carrier_readout_v1"
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Use ignored research results directory")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    if args.phase == "replay":
        dest = out / "numerical_replay"
        joint(out, dest)
        carrier_read(out, dest)
        names = (
            "private_joint_predictions.json",
            "private_joint_models.json",
            "private_carrier_predictions.csv",
        )
        checks = {name: sha(out / name) == sha(dest / name) for name in names}
        write(out / "numerical_replay.json", checks)
        if not all(checks.values()):
            raise ValueError("Numerical replay differed")
        print({"exact_numerical_replays": len(checks)})
    else:
        {
            "prepare": prepare,
            "joint": joint,
            "carrier-collect": carrier_collect,
            "carrier-read": carrier_read,
        }[args.phase](out)


if __name__ == "__main__":
    main()

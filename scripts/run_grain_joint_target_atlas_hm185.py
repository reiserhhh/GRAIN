#!/usr/bin/env python3
"""Construct and audit a complete multi-target atlas, not a prediction grid."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_target_atlas as M
from grain_research.construction import math_target_complement as C
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_joint_target_atlas_hm185.json"
DEFAULT = ROOT / "results/grain_math_response_v1/joint_target_atlas_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_JOINT_TARGET_ATLAS_PLAN_HM185.md",
    "grain_research/construction/math_joint_target_atlas.py",
    "scripts/run_grain_joint_target_atlas_hm185.py",
    "scripts/score_grain_joint_target_atlas_cold.py",
    "tests/test_grain_joint_target_atlas.py",
    "grain_research/construction/math_target_complement.py",
    "scripts/run_grain_observed_family_hm170.py",
)
SPACES = ("full", "retained", "complement")


def prepare(out):
    """Fix source membership,reference and all methods before computing new fields."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    old = R.check(
        parent,
        (
            "compiled_seal.json",
            "scored_seal.json",
            "analysis_seal.json",
            "verification_seal.json",
        ),
    )
    z = R.prepared(parent)
    means, labels = z["source_vectors"].mean(2), z["labels"]
    if means.shape != (386, 2, 1024) or set(z["source_indices"]) & set(
        z["panel_indices"]
    ):
        raise ValueError("Changed source family or source/application overlap")
    codes = M.profile_codes(labels)
    folds = M.author_folds(codes, z["source_indices"], cfg["fold_salt"])
    with np.load(ROOT / cfg["poles"], allow_pickle=False) as p:
        poles = np.asarray(p["query_poles"], float)
    poles = poles / np.linalg.norm(poles, axis=1, keepdims=True)
    if (
        poles.shape != (4, 1024)
        or np.max(np.abs(poles[3] - poles[2] - z["query"]))
        > cfg["comparison_tolerance"]
    ):
        raise ValueError("Historical semantic pole identity changed")
    code = old["code"] | {p: R.digest(ROOT / p) for p in SCIENCE}
    names = (
        "manifest.json",
        "observations.npz",
        "cold_input.jsonl",
        "reader_bundle.npz",
        "prepared_seal.json",
        "compiled_seal.json",
        "verification_seal.json",
    )
    inputs = {str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in names}
    inputs[cfg["poles"]] = R.digest(ROOT / cfg["poles"])
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": code,
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        means=means,
        labels=labels,
        codes=codes,
        folds=folds,
        source_indices=z["source_indices"],
        mapping=z["mapping"],
        query_fields=poles[[1, 3]] - poles[[0, 2]],
    )
    _, counts = M.profile_weights(codes, folds)
    quality = {
        "source_authors": len(codes),
        "fold_authors": np.bincount(folds).tolist(),
        "full_profile_counts": np.bincount(codes, minlength=32).tolist(),
        "fold_profile_counts": counts.tolist(),
        "fold_minimum_profile_count": int(counts.min()),
        "source_application_overlap": 0,
        "native_query_inventory": {
            "TF_authors": 40,
            "N_authors": 24,
            "same_author_overlap": 0,
            "same_event_multi_query": False,
        },
        "reference": cfg["reference"],
        "new_neural_API_private_raw_application_labels": 0,
    }
    R.write(out / "quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "quality.json"))
    print(json.dumps(quality))


def compile_source(out):
    """Build all source fields and a fixed finite exchange reference, without fitting."""
    cfg = R.check(out)["config"]
    z, start = R.prepared(out), time.monotonic()
    basis, _ = C.fixed_basis(z["mapping"])
    pooled = M.construct(z["means"], z["codes"])
    crossed = M.construct(z["means"], z["codes"], z["folds"])
    fields = C.split_fields(pooled["fields"][0], basis)
    fold_fields = C.split_fields(crossed["fields"], basis)
    semantic = C.split_fields(z["query_fields"], basis)
    stats = M.cross_author_statistics(fold_fields)
    rng = np.random.default_rng(cfg["null_seed"])
    nulls = np.empty((cfg["null_replicates"], len(stats)))
    for replicate in range(cfg["null_replicates"]):
        c = M.permute_profiles(z["codes"], z["folds"], rng)
        p = M.construct(z["means"], c, z["folds"])
        f = C.split_fields(p["fields"], basis)
        nulls[replicate] = [
            r["signed_inner_product"] for r in M.cross_author_statistics(f)
        ]
    rows = []
    for i, row in enumerate(stats):
        rows.append(
            row
            | {
                "space": SPACES[row["space"]],
                "null_mean": float(nulls[:, i].mean()),
                "null_min": float(nulls[:, i].min()),
                "null_max": float(nulls[:, i].max()),
                "upper_tail": float(
                    (1 + np.count_nonzero(nulls[:, i] >= row["signed_inner_product"]))
                    / (len(nulls) + 1)
                ),
            }
        )
    modes = []
    for mask in range(32):
        for s, space in enumerate(SPACES):
            a, b = fold_fields[0, :, mask, s], fold_fields[1, :, mask, s]
            dot = float(a.mean(0) @ b.mean(0))
            den = np.linalg.norm(a) * np.linalg.norm(b)
            modes.append(
                {
                    "mask": mask,
                    "targets": "+".join(
                        t for q, t in enumerate(M.TARGETS) if mask & (1 << q)
                    )
                    or "common",
                    "order": mask.bit_count(),
                    "space": space,
                    "signed_inner_product": dot,
                    "cross_family_coherence": float(2 * dot / den) if den else None,
                    "within_fold_view_inner_product": float(
                        (a[0] @ a[1] + b[0] @ b[1]) / 2
                    ),
                }
            )
    edges = np.einsum("qbm,vmsd->vqbsd", M.edge_matrix(), fields)
    cells = pooled["cells"][0]
    direct = np.empty((2, 5, 16, 1024))
    for q in range(5):
        others = [j for j in range(5) if j != q]
        for b in range(16):
            lo = sum(((b >> j) & 1) << k for j, k in enumerate(others))
            direct[:, q, b] = cells[:, lo | (1 << q)] - cells[:, lo]
    R.archive(
        out / "reader_bundle.npz",
        version=np.asarray(M.VERSION),
        fields=fields,
        semantic=semantic,
    )
    R.archive(
        out / "construction_accounts.npz",
        fields=fields,
        fold_fields=fold_fields,
        cells=cells,
        counts=crossed["counts"],
        weights=pooled["weights"][0],
        fold_weights=crossed["weights"],
        null_statistics=nulls,
        edges=edges,
    )
    R.table(out / "order_comparison.csv", rows)
    R.table(out / "all_modes.csv", modes)
    receipt = {
        "source_profiles": 32,
        "modes_including_common": 32,
        "conditional_target_edges_per_view": 80,
        "direct_edge_error": float(np.max(np.abs(direct - edges[:, :, :, 0]))),
        "profile_reconstruction_error": float(
            np.max(
                np.abs(
                    np.einsum("gm,vmd->vgd", M.characters(), pooled["fields"][0])
                    - cells
                )
            )
        ),
        "seconds": time.monotonic() - start,
        "new_neural_API_native_answer_fit": 0,
    }
    if (
        max(receipt["direct_edge_error"], receipt["profile_reconstruction_error"])
        > cfg["comparison_tolerance"]
    ):
        raise ValueError("Incomplete atlas; preserve failure instead of refitting")
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "reader_bundle.npz",
            "construction_accounts.npz",
            "order_comparison.csv",
            "all_modes.csv",
            "construction_receipt.json",
        ),
    )
    print(
        json.dumps(
            {
                "receipt": receipt,
                "primary": next(
                    r for r in rows if r["order"] == "joint" and r["space"] == "full"
                ),
            }
        )
    )


def score(out):
    """Run the separate numeric scorer on all historical frames, not a new sample."""
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_joint_target_atlas_cold.py"),
            "--bundle",
            str(out / "reader_bundle.npz"),
            "--input",
            str(ROOT / cfg["parent"] / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        R.write(
            out / "cold_failure.json",
            {"returncode": proc.returncode, "stderr": proc.stderr},
        )
        raise RuntimeError("Cold scorer failed; no automatic retry")
    receipt = json.loads(proc.stdout)
    if receipt["frames"] != cfg["cold_frames"]:
        raise ValueError("Incomplete cold frame scope")
    R.write(out / "cold_receipt.json", receipt)
    R.seal(out, "scored_seal.json", ("cold_output.jsonl", "cold_receipt.json"))
    print(json.dumps(receipt))


def verify(out):
    """Replay literal observed cells and original factor coordinates, not just tests."""
    cfg = R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"]
    z, saved = (
        R.prepared(out),
        np.load(out / "construction_accounts.npz", allow_pickle=False),
    )
    errors = {
        "literal_profile_means": 0.0,
        "literal_coefficients": 0.0,
        "cold_mode": 0.0,
        "cold_edges": 0.0,
        "cold_semantic": 0.0,
        "original_factor_reading": 0.0,
        "matched_HM184_replay": 0.0,
    }
    for bank in range(2):
        for view in range(2):
            literal = np.array(
                [
                    z["means"][(z["codes"] == g) & (z["folds"] == bank), view].mean(0)
                    for g in range(32)
                ]
            )
            for mask in range(32):
                signs = np.array(
                    [
                        np.prod(
                            [
                                2 * ((g >> q) & 1) - 1
                                for q in range(5)
                                if mask & (1 << q)
                            ]
                        )
                        for g in range(32)
                    ]
                )
                coefficient = sum(signs[g] * literal[g] for g in range(32)) / 32
                errors["literal_coefficients"] = max(
                    errors["literal_coefficients"],
                    float(
                        np.max(
                            np.abs(
                                coefficient - saved["fold_fields"][bank, view, mask, 0]
                            )
                        )
                    ),
                )
    cells = np.array(
        [[z["means"][z["codes"] == g, v].mean(0) for g in range(32)] for v in range(2)]
    )
    errors["literal_profile_means"] = float(np.max(np.abs(cells - saved["cells"])))
    with np.load(ROOT / cfg["parent"] / "reader_bundle.npz", allow_pickle=False) as old:
        original = old["fields"][:, :, 1, 0]
    for q in range(5):
        other = np.delete(z["labels"], q, axis=1) @ (2 ** np.arange(4))
        n = np.array(
            [
                np.bincount(other[z["labels"][:, q] == g].astype(int), minlength=16)
                for g in (0, 1)
            ]
        )
        w = n.prod(0) / n.sum(0)
        w /= w.sum()
        errors["matched_HM184_replay"] = max(
            errors["matched_HM184_replay"],
            float(
                np.max(
                    np.abs(
                        saved["edges"][:, q, :, 0].transpose(0, 2, 1) @ w
                        - original[:, q]
                    )
                )
            ),
        )
    with np.load(out / "reader_bundle.npz", allow_pickle=False) as bundle:
        fields, semantic = bundle["fields"], bundle["semantic"]
    basis, triangular = C.fixed_basis(z["mapping"])
    beta = C.coordinate_fields(fields[:, :, 0], basis, triangular)
    frames = 0
    with (
        (ROOT / cfg["parent"] / "cold_input.jsonl").open() as source,
        (out / "cold_output.jsonl").open() as output,
    ):
        for line, result in zip(source, output, strict=True):
            row, got = json.loads(line), json.loads(result)
            if row["frame_key"] != got["frame_key"]:
                raise ValueError("Changed event frame identity")
            x = np.array([e["vector"] for e in row["events"]])
            literal = np.array(
                [
                    [
                        [sum(float(d @ event) for event in x) / len(x) for d in space]
                        for space in view
                    ]
                    for view in fields
                ]
            )
            errors["cold_mode"] = max(
                errors["cold_mode"], float(np.max(np.abs(literal - got["modes"])))
            )
            literal_edges = np.array(
                [
                    [
                        [
                            (
                                cells[
                                    v,
                                    sum(
                                        ((b >> j) & 1) << k
                                        for j, k in enumerate(
                                            [i for i in range(5) if i != q]
                                        )
                                    )
                                    | (1 << q),
                                ]
                                - cells[
                                    v,
                                    sum(
                                        ((b >> j) & 1) << k
                                        for j, k in enumerate(
                                            [i for i in range(5) if i != q]
                                        )
                                    ),
                                ]
                            )
                            @ x.mean(0)
                            for b in range(16)
                        ]
                        for q in range(5)
                    ]
                    for v in range(2)
                ]
            )
            errors["cold_edges"] = max(
                errors["cold_edges"],
                float(np.max(np.abs(literal_edges - np.asarray(got["edges"])[..., 0]))),
            )
            errors["cold_semantic"] = max(
                errors["cold_semantic"],
                float(
                    np.max(
                        np.abs(
                            np.einsum("qsd,d->qs", semantic, x.mean(0))
                            - got["semantic"]
                        )
                    )
                ),
            )
            coord = np.einsum("vmd,d->vm", beta, (x @ z["mapping"]).mean(0))
            errors["original_factor_reading"] = max(
                errors["original_factor_reading"],
                float(np.max(np.abs(coord - np.asarray(got["modes"])[:, :, 1]))),
            )
            frames += 1
    if (
        frames != cfg["cold_frames"]
        or max(errors.values()) > cfg["comparison_tolerance"]
    ):
        raise ValueError("Failed literal replay; do not adjust science/tolerances")
    R.write(
        out / "verification.json",
        {"frames": frames, "errors": errors, "same_agent_not_blind": True},
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(R.read(out / "verification.json")))


def main():
    """Execute one immutable stage, retaining every scientific failure."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compile", "score", "verify"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = p.parse_args()
    {"prepare": prepare, "compile": compile_source, "score": score, "verify": verify}[
        args.stage
    ](args.output_dir)


if __name__ == "__main__":
    main()

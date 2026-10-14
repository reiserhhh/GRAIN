#!/usr/bin/env python3
"""Supplemental fresh-text GRAIN/native fiber audit; no new neural execution."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_observation_collision import collision_account
from scripts import run_grain_author_evidence_merge_hm102 as H
from suica_core.suica import TOKEN_RE
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle


def prepare(out):
    """Score all48 fresh texts using the inherited frozen GRAIN lexical operator."""
    folder = out / "collision_supplement_r1"
    if folder.exists():
        raise FileExistsError("Preserve all previous supplemental attempts")
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    spec = H.load(ROOT / "configs/grain_marked_response_hm120.json")
    bundle = read_factor_bundle(ROOT / spec["grain_bundle"])
    representation = joblib.load(ROOT / spec["grain_representation"])
    rows, records = H.load(out / "00/rows.json"), []
    for row in rows:
        text = row["text"]
        words = list(TOKEN_RE.finditer(text))
        cuts = [
            0,
            words[len(words) // 3].start(),
            words[2 * len(words) // 3].start(),
            len(text),
        ]
        records.append({"author": f"r{row['id']:03d}_whole", "body": text})
        for j, (a, b) in enumerate(itertools.pairwise(cuts)):
            part = text[a:b]
            if len(TOKEN_RE.findall(part)) < 12:
                raise ValueError("Do not rescue unscorable slices")
            records.append({"author": f"r{row['id']:03d}_s{j}", "body": part})
    frame = canonicalize_comments(
        pd.DataFrame(records), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    _, _, scored = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=representation,
        bundle=bundle,
    )
    if len(scored) != 192 or not scored.score_status.eq("SCORED").all():
        raise ValueError("No refusal may be converted to zero")
    scores = (
        scored.set_index("user_id")
        .loc[
            [r["author"] for r in records],
            [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)],
        ]
        .to_numpy(float)
        .reshape(48, 4, 4)
    )
    folder.mkdir()
    with (folder / "scores.npz").open("xb") as stream:
        np.savez_compressed(
            stream, whole=scores[:, 0], slices=scores[:, 1:].reshape(48, 12)
        )
    code = [
        "scripts/analyze_grain_observation_collision_hm124.py",
        "grain_research/construction/math_observation_collision.py",
        "tests/test_grain_observation_collision.py",
        "suica_core/v7_observations.py",
        "suica_core/v7_operator_controls.py",
        "suica_core/v7_psychometric.py",
    ]
    H.write(
        folder / "manifest.json",
        {
            "code": {n: H.sha(ROOT / n) for n in code},
            "assets": {
                n: H.sha(ROOT / n)
                for n in (spec["grain_bundle"], spec["grain_representation"])
            },
            "inputs": {"00/rows.json": H.sha(out / "00/rows.json")},
            "scores": H.sha(folder / "scores.npz"),
            "supplement_started_after_source_invariance_found": True,
            "supplemental_coordinates_never_used_to_select_native_actions": True,
            "new_neural_calls": 0,
        },
    )
    print("HM124 supplement scored192 whole/slice records; no neural calls", flush=True)


def analyze(out):
    """Compare native baseline response variation inside exact GRAIN fibers."""
    folder = out / "collision_supplement_r1"
    manifest = H.load(folder / "manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["assets"])
    H.verify_hashes(out, manifest["inputs"])
    if H.sha(folder / "scores.npz") != manifest["scores"]:
        raise ValueError("Frozen supplemental scores changed")
    if H.load(out / "status.json")["state"] != "COLLECTED":
        raise ValueError("Native collection must be terminal")
    rows = H.load(out / "00/rows.json")
    native = []
    for row in rows:
        stem = out / f"01/units/b{row['id'] * 5:03d}"
        if (
            H.sha(stem.with_suffix(".npz"))
            != H.load(stem.with_suffix(".json"))["sha256"]
        ):
            raise ValueError("Native baseline changed")
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            native.append(z["readings"].reshape(2, 6, 4))
    native = np.array(native)
    result, arrays = {}, {"native": native}
    with np.load(folder / "scores.npz", allow_pickle=False) as scores:
        for name in scores.files:
            g = scores[name]
            account = collision_account(g, native)
            owner_error, owner_response = 0.0, []
            for i in range(0, 48, 2):
                owner_error = max(owner_error, float(np.max(abs(g[i] - g[i + 1]))))
                owner_response.append(native[i] - native[i + 1])
            difference = np.array(owner_response)
            result[name] = {
                "distinct_observations": account["distinct_observations"],
                "collision_rows": account["collision_rows"],
                "owner_swap_coordinate_max": owner_error,
                "native_owner_difference_rms": np.sqrt(
                    np.mean(difference**2, axis=(0, 1, 2))
                ).tolist(),
                "irreducible_rmse_by_focus_target_unit": np.sqrt(
                    account["irreducible_mse"]
                ).tolist(),
                "source_of_bound": "Exact finite uniform-ledger equivalence classes; not population generalization error",
            }
            arrays.update(
                {
                    name + "_" + k: v
                    for k, v in account.items()
                    if isinstance(v, np.ndarray)
                }
            )
    with (folder / "account.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    H.write(
        folder / "summary.json",
        {
            "status": "FINITE_OBSERVATION_FIBERS_ACCOUNTED_NOT_FULL",
            "coordinates": result,
            "current_lexical_operator_only": True,
            "no_GRAIN_wide_or_human_personality_claim": True,
            "arbitrary_nonlinear_readout_cannot_split_exact_identical_inputs": True,
        },
    )
    H.write(folder / "seal.json", H.seals(out, folder))
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "analyze"))
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/grain_native_remainder_r1",
    )
    a = p.parse_args()
    {"prepare": prepare, "analyze": analyze}[a.phase](a.out.resolve())

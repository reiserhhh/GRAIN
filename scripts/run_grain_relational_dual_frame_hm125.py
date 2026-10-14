#!/usr/bin/env python3
"""Construct and evaluate one source-fixed vector ruler; no model execution."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_relational_dual_frame as M
from scripts import run_grain_author_evidence_merge_hm102 as H

SPEC = ROOT / "configs/grain_relational_dual_frame_hm125.json"
CODE = (
    "configs/grain_relational_dual_frame_hm125.json",
    "docs/GRAIN_RELATIONAL_DUAL_FRAME_PLAN_HM125.md",
    "grain_research/construction/math_relational_dual_frame.py",
    "scripts/run_grain_relational_dual_frame_hm125.py",
    "scripts/readback_grain_relational_dual_frame_hm125.py",
    "scripts/score_grain_relational_dual_frame_cold.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "tests/test_grain_relational_dual_frame.py",
)
VIEWS = ("source", "same_event_rewording", "different_instance", "fresh_instance")


def save(path, **arrays):
    """Preserve earlier and failed outputs by exclusively creating each unit."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def budget_expired(_signum, _frame):
    """Abort arithmetic at the original nonrenewable process-CPU allowance."""
    raise TimeoutError("Original HM125 CPU budget exhausted")


@contextlib.contextmanager
def budget(out, phase):
    """Record cumulative CPU use and preserve a terminal failed checkpoint."""
    state = H.load(out / "status.json")
    remaining = H.load(SPEC)["max_cpu_seconds"] - state["cpu_seconds"]
    if remaining <= 0 or state["state"] == "FAILED":
        raise RuntimeError("No renewal of a failed or exhausted packet")
    before = time.process_time()
    previous = signal.signal(signal.SIGPROF, budget_expired)
    signal.setitimer(signal.ITIMER_PROF, remaining)
    try:
        yield state
    except BaseException as error:
        state.update(state="FAILED", failed_phase=phase, error=repr(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)
        state["cpu_seconds"] += time.process_time() - before
        state["last_phase"] = phase
        with (out / "status.json").open("w") as stream:
            import json

            json.dump(state, stream, indent=2, allow_nan=False)


def check(out):
    """Verify the construction and the frozen provenance, not native answers."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def profiles(spec):
    """Return the fixed neutral/axis/dense finite virtual-author laws."""
    amplitude = spec["profile_amplitude"]
    theta, pairs = [np.zeros((6, 2))], []
    rng = np.random.default_rng(spec["profile_seed"])
    seeds = [amplitude * np.eye(12)[j].reshape(6, 2) for j in range(12)]
    seeds += list(
        rng.uniform(-amplitude, amplitude, (spec["dense_profile_pairs"], 6, 2))
    )
    for t in seeds:
        pairs.append((len(theta), len(theta) + 1))
        theta.extend((t, -t))
    return np.array(theta), np.array(pairs, int)


def prepare(out):
    """Fit only original96 source vectors before admitting application views."""
    if out.exists():
        raise FileExistsError("Do not replace earlier or failed runs")
    (out / "00").mkdir(parents=True)
    H.write(out / "status.json", {"state": "PREPARING", "cpu_seconds": 0.0})
    with budget(out, "prepare") as state:
        spec = H.load(SPEC)
        source, fresh = ROOT / spec["source"], ROOT / spec["fresh"]
        H.verify_hashes(source, H.load(source / "02/output_seal.json"))
        old_manifest = H.load(fresh / "00/manifest.json")
        H.verify_hashes(ROOT, old_manifest["code"] | old_manifest["source"])
        H.verify_hashes(fresh, old_manifest["inputs"])
        rows = [r for r in H.load(source / "00/rows.json") if r["variant"] == 0]
        with np.load(source / "02/fields.npz", allow_pickle=False) as z:
            x = z["v"][0].reshape(spec["source_records"], -1).astype(float)
        cells = list(itertools.product(range(6), range(2)))
        d = M.action_columns(x, rows, cells)
        with np.load(fresh / "00/decomposition.npz", allow_pickle=False) as z:
            old_d = z["whole"].reshape(12, -1).T
        if np.max(abs(d - old_d)) > spec["numeric_atol"]:
            raise ValueError("Source actor columns disagree with frozen HM124")
        frame = M.fit_frame(x, d)
        save(out / "00/frame.npz", **frame)
        save(out / "00/source_vectors.npz", vectors=x)
        theta, pairs = profiles(spec)
        save(out / "00/profiles.npz", theta=theta, pairs=pairs)
        H.write(out / "00/source_rows.json", rows)
        names = [
            source / "00/rows.json",
            source / "02/fields.npz",
            source / "02/output_seal.json",
            ROOT / spec["marks"],
            fresh / "00/rows.json",
            fresh / "00/decomposition.npz",
            fresh / "00/manifest.json",
            fresh / "status.json",
            fresh / "collision_supplement_r1/scores.npz",
            fresh / "collision_supplement_r1/manifest.json",
        ]
        names += [
            p
            for p in sorted((fresh / "01/units").glob("b*.json"))
            if int(p.stem[1:]) % 5 == 0
        ]
        H.write(
            out / "00/manifest.json",
            {
                "spec": spec,
                "code": {n: H.sha(ROOT / n) for n in CODE},
                "source": {str(p.relative_to(ROOT)): H.sha(p) for p in names},
                "inputs": H.seals(out, out / "00"),
                "fit_vectors": 96,
                "fit_uses_answers": False,
                "application_metadata_enters_scorer": False,
                "encoder_is_frozen_pretrained_not_LLM_free": True,
                "original_scope_complete": False,
            },
        )
        state["state"] = "PREPARED"
    print("HM125 source-only dual sealed", flush=True)


def application(out):
    """Apply the one ruler; source coordinates, residuals and margins all stay."""
    if H.load(out / "status.json")["state"] != "PREPARED":
        raise RuntimeError("One execution only; no terminal replay")
    with budget(out, "application") as state:
        spec = check(out)["spec"]
        source, fresh = ROOT / spec["source"], ROOT / spec["fresh"]
        with np.load(out / "00/frame.npz", allow_pickle=False) as z:
            frame = dict(z)
        with np.load(out / "00/profiles.npz", allow_pickle=False) as z:
            theta, pairs = z["theta"], z["pairs"]
        rows = H.load(source / "00/rows.json")
        with np.load(source / "02/fields.npz", allow_pickle=False) as z:
            vectors = z["v"].reshape(3, 96, -1).astype(float)
        with np.load(ROOT / spec["marks"], allow_pickle=False) as z:
            lexical = z["slice_scores"].reshape(3, 96, 12)
        fresh_rows = H.load(fresh / "00/rows.json")
        fresh_vectors, cut_error = [], 0.0
        for r in fresh_rows:
            stem = fresh / f"01/units/b{5 * r['id']:03d}"
            if (
                H.sha(stem.with_suffix(".npz"))
                != H.load(stem.with_suffix(".json"))["sha256"]
            ):
                raise ValueError("Baseline pre-edit cache changed")
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
                before = z["before"].reshape(12, -1).astype(float)
            cut_error = max(cut_error, float(np.max(abs(before - before[0]))))
            fresh_vectors.append(before[0])
        with np.load(
            fresh / "collision_supplement_r1/scores.npz", allow_pickle=False
        ) as z:
            fresh_g = z["slices"]
        if cut_error > spec["native_causal_cut_atol"]:
            raise ValueError("Future-question-independent vector cut failed")
        (out / "01").mkdir()
        summaries, row_addresses = {}, {}
        for v, name in enumerate(VIEWS):
            bank = [r for r in rows if r["variant"] == v] if v < 3 else fresh_rows
            x, g = (
                (vectors[v], lexical[v])
                if v < 3
                else (np.array(fresh_vectors), fresh_g)
            )
            cells = (
                list(itertools.product(range(6), range(2)))
                if v < 3
                else [(f, 0) for f in range(6)]
            )
            scored = M.score_vectors(x, frame)
            p = M.make_profile_laws(bank, theta, cells)
            author = M.compose_profiles(p, scored | {"raw_vectors": x})
            d = M.action_columns(x, bank, cells)
            transport = frame["dual"] @ d
            indices = [2 * f + c for f, c in cells]
            contrast = (
                len(cells)
                * (
                    author["coordinates"][pairs[:, 0]]
                    - author["coordinates"][pairs[:, 1]]
                )
                / 2
            )
            expected = (
                theta[pairs[:, 0]].reshape(len(pairs), 12)[:, indices] @ transport.T
            )
            lexical_error, mass = M.lexical_law_distance(g, p)
            captured = frame["columns"] @ transport
            diagonal = np.array([transport[j, k] for k, j in enumerate(indices)])
            origin = (x.mean(0) - frame["center"]) @ frame["dual"].T
            off_diagonal = transport.copy()
            off_diagonal[indices, np.arange(len(indices))] = 0
            save(
                out / f"01/{name}.npz",
                vectors=x,
                lexical=g,
                laws=p,
                coordinates=scored["coordinates"],
                complement=scored["complement"],
                author_coordinates=author["coordinates"],
                author_complement=author["complement"],
                author_raw_vectors=author["raw_vectors"],
                action_columns=d,
                transport=transport,
                contrasts=contrast,
                expected=expected,
                class_mass=mass,
                visible_indices=np.array(indices),
                origin=origin,
            )
            summaries[name] = {
                "records": len(x),
                "profiles": len(p),
                "covered_cells": len(cells),
                "contrast_identity_max_error": float(np.max(abs(contrast - expected))),
                "lexical_entire_law_max_change": lexical_error,
                "source_frame_capture_energy_fraction": float(
                    np.sum(captured**2) / np.sum(d**2)
                ),
                "transport_singular_values": np.linalg.svd(
                    transport, compute_uv=False
                ).tolist(),
                "transport_diagonal": diagonal.tolist(),
                "positive_diagonal_count": int(np.sum(diagonal > 0)),
                "off_diagonal_to_diagonal_norm": float(
                    np.linalg.norm(off_diagonal) / np.linalg.norm(diagonal)
                ),
                "source_reference_origin": origin.tolist(),
                "source_reference_origin_norm": float(np.linalg.norm(origin)),
                "maximum_profile_coordinate_distance": float(
                    np.max(
                        np.linalg.norm(
                            author["coordinates"][:, None]
                            - author["coordinates"][None],
                            axis=2,
                        )
                    )
                ),
                "unobserved_conditions": [1] if v == 3 else [],
            }
            row_addresses[name] = [
                {
                    k: r[k]
                    for k in (
                        "id",
                        "wording",
                        "family",
                        "condition",
                        "polarity",
                        "owner",
                    )
                }
                for r in bank
            ]
            if (
                max(lexical_error, 0) > spec["lexical_law_atol"]
                or np.max(abs(contrast - expected)) > spec["numeric_atol"]
            ):
                raise ValueError(
                    "Exact-law/numeric account failed; no threshold rescue"
                )
        H.write(out / "01/rows.json", row_addresses)
        H.write(
            out / "summary.json",
            {
                "status": "CONSTRUCTED_FINITE_DUAL_NOT_FULL_THEORY",
                "new_model_calls": 0,
                "source_rank": 12,
                "source_condition_number": float(frame["condition"]),
                "source_inverse_operator_norm": float(frame["inverse_norm"]),
                "future_question_preedit_vector_max_error": cut_error,
                "future_question_coordinate_error_bound": cut_error
                * np.sqrt(len(frame["center"]))
                * float(frame["inverse_norm"]),
                "views": summaries,
                "source_identity_is_calibration_not_discovery": True,
                "application_no_refit_no_native_answer_no_design_input": True,
                "virtual_profiles_not_human_traits": True,
                "current_V_encoder_remains_pretrained_contextual_model": True,
                "original_scope_complete": False,
            },
        )
        H.write(
            out / "output_seal.json",
            H.seals(out, out / "01") | {"summary.json": H.sha(out / "summary.json")},
        )
        state["state"] = "COLLECTED"
    print("HM125 scored336 vectors and260 virtual laws; zero model calls", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "application"))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/relational_dual_frame_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "application": application}[args.phase](args.out.resolve())

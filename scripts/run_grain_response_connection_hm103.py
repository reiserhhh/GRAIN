#!/usr/bin/env python3
"""Execute response geometry on sealed public carriers, without a predictor fit."""

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
from grain_research.construction import math_response_connection as M
from scripts import run_grain_author_evidence_merge_hm102 as H

SPEC = ROOT / "configs/grain_response_connection_hm103.json"
CODE = (
    "configs/grain_response_connection_hm103.json",
    "docs/GRAIN_RESPONSE_CONNECTION_PLAN_HM103.md",
    "grain_research/construction/math_response_connection.py",
    "scripts/run_grain_response_connection_hm103.py",
    "tests/test_grain_math_response_connection.py",
)
MODES = ("joint", "value_only")


def stat(a):
    """Report signed values and magnitudes, without selecting favorable cases."""
    a = np.asarray(a, float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "rms": float(np.sqrt(np.mean(a**2))),
    }


def models(reader, carriers, row, mode):
    """Construct the declared battery and independent source-mean probe."""
    start, end = carriers["offsets"][row : row + 2]
    k, v = carriers["keys"][start:end], carriers["values"][start:end]
    q = np.asarray(reader["positions"][str(carriers["endpoints"][row])]["queries"])
    d = np.asarray(reader["directions"])
    return (
        M.AttentionResponse(k, v, q, d, mode),
        M.AttentionResponse(
            k, v, q.mean(0, keepdims=True), d.mean(0, keepdims=True), mode
        ),
    )


def run(out):
    """Freeze lineage, calculate all brackets, then execute the fixed finite loops."""
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    H.check(source)
    H.verify_hashes(source, H.load(source / "output_seal.json"))
    H.verify_hashes(source, H.load(source / "interface_seal.json")["outputs"])
    if out.exists():
        raise FileExistsError("Retain every prior partial or failed result")
    out.mkdir(parents=True)
    files = [
        "00/reader.json",
        "00/carriers.npz",
        "00/manifest.json",
        "verification.json",
    ]
    manifest = {
        "code": {name: H.sha(ROOT / name) for name in CODE},
        "source": {
            str((source / name).relative_to(ROOT)): H.sha(source / name)
            for name in files
        },
        "source_verified_by_original_checker": True,
        "scope": "off-manifold numeric response controls, not human or natural text loops",
    }
    H.write(out / "manifest.json", manifest)
    reader = H.load(source / "00/reader.json")
    carriers = np.load(source / "00/carriers.npz", allow_pickle=False)
    indices = carriers["indices"]
    pairs = list(itertools.combinations(range(6), 2))
    probe_actions = np.full((2, len(indices), len(pairs)), np.nan)
    bracket_rows, loop_rows, failed, witnesses = [], [], [], []
    start = time.monotonic()

    def deadline():
        """Stop at the fixed budget with all finished scientific rows preserved."""
        if time.monotonic() - start > spec["cpu_deadline_seconds"]:
            raise TimeoutError("Fixed CPU packet deadline; no automatic replay")

    try:
        for row, index in enumerate(indices):
            deadline()
            meta = dict(
                zip(
                    ("bank", "bridge", "family", "previous", "current", "arm"),
                    map(int, index),
                    strict=True,
                )
            )
            for mode_id, mode in enumerate(MODES):
                model, probe = models(reader, carriers, row, mode)
                try:
                    f, j, u, c, declared_pairs, condition = model.brackets(
                        model.x, spec["gram_rcond"]
                    )
                    if declared_pairs != pairs:
                        raise ValueError("Fixed six-target pair support changed")
                    _, gp = probe.value_jacobian(probe.x)
                    probe_actions[mode_id, row] = (gp @ c.T).ravel()
                    for pair_id, (i, k) in enumerate(pairs):
                        bracket_rows.append(
                            {
                                "row": row,
                                **meta,
                                "mode": mode,
                                "i": i,
                                "j": k,
                                "norm": float(np.linalg.norm(c[pair_id])),
                                "vertical_error": float(np.max(np.abs(j @ c[pair_id]))),
                                "dual_error": float(np.max(np.abs(j @ u - np.eye(6)))),
                                "probe_action": float(
                                    probe_actions[mode_id, row, pair_id]
                                ),
                                "jacobian_condition": condition,
                            }
                        )
                    if not (index[3] == index[4] == 1 and index[5] in (0, 3)):
                        continue
                    p0 = probe.value_jacobian(probe.x)[0][0]
                    for pair_id, (i, k) in enumerate(pairs):
                        for label in spec["loop_doses"]:
                            deadline()
                            dose, sign = abs(label), 1 if label > 0 else -1
                            a, b = (i, k) if sign == 1 else (k, i)
                            y = M.rectangle(
                                model,
                                model.x,
                                a,
                                b,
                                dose,
                                spec["rk4_steps_per_leg"],
                                spec["gram_rcond"],
                            )
                            delta = y - model.x
                            predicted = sign * dose**2 * c[pair_id]
                            fend = model.value_jacobian(y)[0]
                            pend = probe.value_jacobian(y)[0][0]
                            denominator = max(float(np.linalg.norm(predicted)), 1e-14)
                            loop_rows.append(
                                {
                                    "row": row,
                                    **meta,
                                    "mode": mode,
                                    "i": i,
                                    "j": k,
                                    "label": label,
                                    "dose": dose,
                                    "orientation": sign,
                                    "drift_norm": float(np.linalg.norm(delta)),
                                    "predicted_norm": float(np.linalg.norm(predicted)),
                                    "bracket_relative_error": float(
                                        np.linalg.norm(delta - predicted) / denominator
                                    ),
                                    "six_score_closure_max": float(
                                        np.max(np.abs(fend - f))
                                    ),
                                    "probe_delta": float(pend - p0),
                                    "probe_prediction": float(
                                        sign
                                        * dose**2
                                        * probe_actions[mode_id, row, pair_id]
                                    ),
                                }
                            )
                            if mode == "joint" and label == spec["loop_doses"][0]:
                                witnesses.append(
                                    (
                                        row,
                                        pair_id,
                                        model.x.copy(),
                                        delta.copy(),
                                        predicted.copy(),
                                    )
                                )
                except ValueError as exc:
                    failed.append({"row": row, **meta, "mode": mode, "error": str(exc)})
            if row % 96 == 0:
                print(
                    json.dumps(
                        {
                            "rows_finished": row + 1,
                            "brackets": len(bracket_rows),
                            "loops": len(loop_rows),
                            "seconds": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
    except (TimeoutError, KeyboardInterrupt) as exc:
        failed.append({"error": str(exc), "terminal": "partial_not_replayed"})
    finally:
        if bracket_rows:
            H.table(out / "brackets.csv", bracket_rows)
        if loop_rows:
            H.table(out / "loops.csv", loop_rows)
        H.write(out / "failures.json", failed)
    np.savez_compressed(
        out / "probe_actions.npz", probe_actions=probe_actions, indices=indices
    )
    offsets = [0]
    for _, _, x, _, _ in witnesses:
        offsets.append(offsets[-1] + len(x))
    if witnesses:
        np.savez_compressed(
            out / "loop_witnesses.npz",
            rows=[w[0] for w in witnesses],
            pairs=[w[1] for w in witnesses],
            offsets=offsets,
            base=np.concatenate([w[2] for w in witnesses]),
            delta=np.concatenate([w[3] for w in witnesses]),
            prediction=np.concatenate([w[4] for w in witnesses]),
        )
    profile_rows = []
    pplus, pminus = H.M.author_laws()
    for group in sorted({tuple(i[[0, 1, 2, 5]]) for i in indices}):
        rows = np.flatnonzero(np.all(indices[:, [0, 1, 2, 5]] == group, axis=1))
        if len(rows) != 9 or not np.array_equal(
            indices[rows][:, 3:5], list(itertools.product(range(3), repeat=2))
        ):
            raise ValueError("Exact declared nine-record profile required")
        for mode_id, mode in enumerate(MODES):
            vals = probe_actions[mode_id, rows]
            if not np.isfinite(vals).all():
                continue
            for pair_id, (i, k) in enumerate(pairs):
                plus, minus = (
                    float(pplus @ vals[:, pair_id]),
                    float(pminus @ vals[:, pair_id]),
                )
                profile_rows.append(
                    {
                        "bank": int(group[0]),
                        "bridge": int(group[1]),
                        "family": int(group[2]),
                        "arm": int(group[3]),
                        "mode": mode,
                        "i": i,
                        "j": k,
                        "plus": plus,
                        "minus": minus,
                        "contrast": plus - minus,
                    }
                )
    if profile_rows:
        H.table(out / "profile_curvature.csv", profile_rows)
    summaries = {}
    for mode in MODES:
        br = [r for r in bracket_rows if r["mode"] == mode]
        lr = [r for r in loop_rows if r["mode"] == mode]
        summaries[mode] = {
            "bracket_norm": stat([r["norm"] for r in br]),
            "vertical_error_max": max(r["vertical_error"] for r in br),
            "dual_error_max": max(r["dual_error"] for r in br),
            "probe_action": stat([r["probe_action"] for r in br]),
            "condition": stat([r["jacobian_condition"] for r in br]),
            "finite_drift": stat([r["drift_norm"] for r in lr]),
            "closure_max": max(r["six_score_closure_max"] for r in lr),
            "bracket_relative_error": stat([r["bracket_relative_error"] for r in lr]),
            "finite_probe_delta": stat([r["probe_delta"] for r in lr]),
            "profile_contrast": {
                str(arm): stat(
                    [
                        r["contrast"]
                        for r in profile_rows
                        if r["mode"] == mode and r["arm"] == arm
                    ]
                )
                for arm in range(4)
            },
        }
    summary = {
        "packet": "HM103",
        "status": "COMPUTED_FINITE_NOT_GLOBAL"
        if not failed
        else "PARTIAL_WITH_FAILURES",
        "seconds": time.monotonic() - start,
        "bracket_rows": len(bracket_rows),
        "loop_rows": len(loop_rows),
        "profile_rows": len(profile_rows),
        "failures": len(failed),
        "modes": summaries,
        "new_neural_calls": 0,
        "fits": 0,
        "metric_and_controller_dependent": True,
        "natural_text_or_psychological_loop_claim": False,
        "original_scope_complete": False,
    }
    H.write(out / "summary.json", summary)
    H.write(out / "output_seal.json", H.seals(out, out))
    print(json.dumps(summary, indent=2), flush=True)


def verify(out):
    """Independently check directional derivatives, witnesses and all recorded closure."""
    manifest, spec = H.load(out / "manifest.json"), H.load(SPEC)
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    source = ROOT / spec["source"]
    reader, carriers = (
        H.load(source / "00/reader.json"),
        np.load(source / "00/carriers.npz", allow_pickle=False),
    )
    derivative_error, bracket_error, drift_error, gauge_error = [], [], [], []
    rng = np.random.default_rng(103)
    for row in (0, 173, 346, 519, 692, 863):
        model, _ = models(reader, carriers, row, "joint")
        x = model.x
        _, _j, u, c, _, _ = model.brackets(x)
        a = rng.normal(size=len(x))
        a /= np.linalg.norm(a)
        h = 1e-4
        _, jp = model.value_jacobian(x + h * a)
        _, jm = model.value_jacobian(x - h * a)
        derivative_error.append(
            float(
                np.max(np.abs((jp - jm) / (2 * h) - model.directional_jacobian(x, a)))
            )
        )
        du = []
        for axis in (0, 1):
            hp = h / max(1, np.linalg.norm(u[:, axis]))
            du.append(
                (
                    model.connection(x + hp * u[:, axis])[2]
                    - model.connection(x - hp * u[:, axis])[2]
                )
                / (2 * hp)
            )
        bracket_error.append(float(np.max(np.abs(du[0][:, 1] - du[1][:, 0] - c[0]))))
        n = M.AttentionResponse(
            model.k + rng.normal(size=model.k.shape[1:]), model.v, model.q, model.d
        )
        gauge_error.append(float(np.max(np.abs(n.brackets(n.x)[3] - c))))
    w = np.load(out / "loop_witnesses.npz", allow_pickle=False)
    for row, pair, first, last in zip(
        w["rows"], w["pairs"], w["offsets"][:-1], w["offsets"][1:], strict=True
    ):
        model, _ = models(reader, carriers, int(row), "joint")
        x, delta = w["base"][first:last], w["delta"][first:last]
        if not np.array_equal(x, model.x):
            raise ValueError("Saved loop witness does not start at actual carrier")
        c = model.brackets(x)[3][pair]
        drift_error.append(
            float(
                np.max(
                    np.abs(w["prediction"][first:last] - spec["loop_doses"][0] ** 2 * c)
                )
            )
        )
        drift_error.append(
            float(
                np.max(
                    np.abs(
                        model.value_jacobian(x + delta)[0] - model.value_jacobian(x)[0]
                    )
                )
            )
        )
    summary = H.load(out / "summary.json")
    checks = {
        "hessian_central_error_max": max(derivative_error),
        "bracket_central_error_max": max(bracket_error),
        "gauge_bracket_error_max": max(gauge_error),
        "witness_prediction_and_closure_error_max": max(drift_error),
        "joint_dual_error_max": summary["modes"]["joint"]["dual_error_max"],
        "joint_vertical_error_max": summary["modes"]["joint"]["vertical_error_max"],
        "all_loop_closure_max": max(
            s["closure_max"] for s in summary["modes"].values()
        ),
        "value_only_bracket_norm_max": summary["modes"]["value_only"]["bracket_norm"][
            "max"
        ],
        "neither_author_contrast_max": max(
            abs(summary["modes"]["joint"]["profile_contrast"]["0"][a])
            for a in ("min", "max")
        ),
    }
    passed = (
        max(checks.values()) < spec["numeric_atol"]
        and summary["failures"] == 0
        and summary["bracket_rows"] == 25920
        and summary["loop_rows"] == 4320
        and summary["profile_rows"] == 2880
    )
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM103_NOT_GLOBAL"
            if passed
            else "FAILED_NOT_REPAIRED",
            "checks": checks,
            "same_agent_not_third_party": True,
            "source_code_output_hashes": "PASS",
        },
    )
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    if not passed:
        raise ValueError("Finite numerical verification failed; retained")


def main():
    """Execute or verify an explicitly named fresh scientific packet directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / H.load(SPEC)["output"])
    args = parser.parse_args()
    (run if args.stage == "run" else verify)(args.output)


if __name__ == "__main__":
    main()

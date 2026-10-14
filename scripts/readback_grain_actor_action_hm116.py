#!/usr/bin/env python3
"""Read committed HM116 partial evidence without extending its frozen experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_actor_action_hm116 as R


def characters(cells):
    """Independently integrate all eight c/b/o characters on complete support."""
    x = np.asarray(cells, float)
    if x.shape[:3] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite c/b/o support required")
    result = np.zeros((8, *x.shape[3:]))
    for mask in range(8):
        for c in range(2):
            for b in range(2):
                for o in range(2):
                    sign = np.prod(
                        [
                            (2 * v - 1) if mask & bit else 1
                            for bit, v in ((1, c), (2, b), (4, o))
                        ]
                    )
                    result[mask] += sign * x[c, b, o] / 8
    return result


def complete_groups(native):
    """List only full eight-cell groups; missing observations are never zero."""
    return [
        (a, w, f)
        for a in range(5)
        for w in range(2)
        for f in range(6)
        if np.isfinite(native[a, w, f]).all()
    ]


def run():
    """Authenticate partial captures and save scoped effects and missing cells."""
    out = ROOT / R.H.load(R.SPEC)["output"]
    m, status = R.check(out), R.H.load(out / "status.json")
    if status["state"] not in ("BUDGET_STOP", "FAILED") or status["pending"]:
        raise ValueError("Readback requires a terminal checkpoint without uncertainty")
    design = R.H.load(out / "00/design.json")
    rows = {r["id"]: r for r in R.H.load(out / "00/rows.json")}
    native = np.full((5, 2, 6, 2, 2, 2, 2, 6, 4), np.nan)
    baselines = {d["row"]: d["batch"] for d in design if d["arm"] == "baseline"}
    errors = dict.fromkeys(("key", "before", "edit", "requested", "readout"), 0.0)
    with np.load(out / "00/actions.npz", allow_pickle=False) as z:
        actions = {k: z[k] for k in z.files}
    completed = design[: status["completed"]]
    for d in completed:
        r = rows[d["row"]]
        raw, base = R.raw_unit(out, d["batch"]), R.raw_unit(out, baselines[d["row"]])
        if raw["readings"].shape != (12, 4) or not np.array_equal(
            raw["masked_blocks"], np.tile(np.arange(m["model_blocks"]), (12, 1))
        ):
            raise ValueError("Incomplete endpoint or causal-mask trace")
        for name in (
            "support_hash",
            "execution_support_hash",
            "masked_blocks",
            "execution_length",
            "endpoint",
        ):
            if not np.array_equal(raw[name], base[name]):
                raise ValueError("Execution support changed")
        for name in ("key", "before"):
            field = "k" if name == "key" else name
            errors[name] = max(
                errors[name], float(np.max(abs(raw[field] - base[field])))
            )
        requested = np.zeros(m["physical_shape"])
        if d["arm"] in actions:
            requested = (
                -r["polarity"]
                * r["owner"]
                * actions[d["arm"]][r["family"], (r["condition"] + 1) // 2]
            )
        errors["requested"] = max(
            errors["requested"], float(np.max(abs(requested - raw["requested_delta"])))
        )
        errors["edit"] = max(
            errors["edit"],
            float(np.max(abs(raw["after"] - raw["before"] - raw["requested_delta"]))),
        )
        replay = np.array(
            [
                R.V.native_units(z, n)
                for z, n in zip(raw["logits"], raw["normalizer"], strict=True)
            ]
        )
        errors["readout"] = max(
            errors["readout"], float(np.max(abs(replay - raw["readings"])))
        )
        if d["arm"] == "noop":
            continue
        idx = (
            R.M.ARMS.index(d["arm"]),
            r["wording"],
            r["family"],
            (r["condition"] + 1) // 2,
            (r["polarity"] + 1) // 2,
            (r["owner"] + 1) // 2,
        )
        native[idx] = raw["readings"].reshape(2, 6, 4)
    if (
        errors["edit"] > m["spec"]["edit_tolerance"]
        or max(errors[k] for k in ("key", "before")) > m["spec"]["same_shape_tolerance"]
        or max(errors[k] for k in ("requested", "readout"))
        > m["spec"]["arithmetic_tolerance"]
    ):
        raise ValueError("Observed-capture isolation failed; preserve failure")
    beta = np.full((5, 2, 6, 8, 2, 6, 4), np.nan)
    groups = complete_groups(native)
    author_rows, target_rows, summaries = [], [], []
    author_error = 0.0
    for a, w, f in groups:
        beta[a, w, f] = characters(native[a, w, f])
        integral = R.M.author_difference(native[a, w, f], m["spec"]["author_amplitude"])
        author_error = max(
            author_error,
            float(
                np.max(
                    abs(
                        integral
                        - (2 * m["spec"]["author_amplitude"] / 6) * beta[a, w, f, 6]
                    )
                )
            ),
        )
        for p, person in enumerate(("A", "B")):
            for q, target in enumerate(m["spec"]["targets"]):
                for u, unit in enumerate(R.UNITS):
                    author_rows.append(
                        {
                            "arm": R.M.ARMS[a],
                            "wording": w,
                            "family": m["spec"]["targets"][f],
                            "person": person,
                            "query": target,
                            "unit": unit,
                            "positive_law_difference": float(integral[p, q, u]),
                        }
                    )
    effects = beta[0:1] - beta
    diff = (effects[..., 0, :, :] - effects[..., 1, :, :]) / 2
    common = (effects[..., 0, :, :] + effects[..., 1, :, :]) / 2
    for a, w, f in groups:
        for c, sign in enumerate((-1, 1)):
            for q, target in enumerate(m["spec"]["targets"]):
                for u, unit in enumerate(R.UNITS):
                    target_rows.append(
                        {
                            "arm": R.M.ARMS[a],
                            "wording": w,
                            "family": m["spec"]["targets"][f],
                            "condition": sign,
                            "query": target,
                            "unit": unit,
                            "actor_differential_bo": float(
                                diff[a, w, f, 6, q, u] + sign * diff[a, w, f, 7, q, u]
                            ),
                            "actor_common_bo": float(
                                common[a, w, f, 6, q, u]
                                + sign * common[a, w, f, 7, q, u]
                            ),
                        }
                    )
    for a in range(5):
        for w in range(2):
            if not np.isfinite(diff[a, w]).all():
                continue
            for u, unit in enumerate(R.UNITS):
                x, g = diff[a, w, :, 6, :, u], common[a, w, :, 6, :, u]
                summaries.append(
                    {
                        "arm": R.M.ARMS[a],
                        "wording": w,
                        "unit": unit,
                        "actor_difference_RMS": float(np.sqrt(np.mean(x**2))),
                        "actor_common_RMS": float(np.sqrt(np.mean(g**2))),
                        "diagonal_mean": float(np.trace(x) / 6),
                        "off_target_mean": float(x[~np.eye(6, dtype=bool)].mean()),
                        "diagonal_minus_off": float(
                            np.trace(x) / 6 - x[~np.eye(6, dtype=bool)].mean()
                        ),
                    }
                )
    folder = out / "partial_readback"
    folder.mkdir()
    np.savez_compressed(
        folder / "accounts.npz",
        raw=native,
        beta=beta,
        effects=effects,
        actor_difference=diff,
        actor_common=common,
    )
    R.H.table(folder / "target_effects.csv", target_rows)
    R.H.table(folder / "author_accounts.csv", author_rows)
    R.H.table(folder / "action_summaries.csv", summaries)
    R.H.write(
        folder / "summary.json",
        {
            "status": "VERIFIED_COMMITTED_PARTIAL_NOT_FULL_HM116",
            "checkpoint": status,
            "available_batches": len(completed),
            "missing_batches": design[status["completed"] :],
            "complete_groups": [list(g) for g in groups],
            "checks": errors,
            "independent_author_integral_error": author_error,
            "no_op_available": sum(d["arm"] == "noop" for d in completed),
            "no_op_gate_passed": None,
            "all_character_coordinates": 8,
            "target_rows": len(target_rows),
            "author_rows": len(author_rows),
            "neural_runtime_loaded": any(
                n in sys.modules for n in ("torch", "transformers")
            ),
            "companion_code_sha256": R.H.sha(Path(__file__)),
            "full_protocol_passed": False,
            "new_forwards": 0,
            "new_fits": 0,
            "collection_checkpoint_unchanged": True,
            "original_scope_complete": False,
        },
    )
    R.H.write(folder / "output_seal.json", R.H.seals(out, folder))
    print(
        "PARTIAL READBACK",
        len(completed),
        "batches",
        len(groups),
        "complete groups",
        errors,
        "author integral",
        author_error,
        flush=True,
    )


if __name__ == "__main__":
    run()

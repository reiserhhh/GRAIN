#!/usr/bin/env python3
"""HM31 alternate score replay, exact rational coordinates and product projections."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import alternate_parts, table
from scripts.verify_grain_math_position_support import rational_inverse, score

PARENT = ROOT / "results/grain_math_response_v1/context_fiber_recombination_r1"
OUT = ROOT / "results/grain_math_response_v1/context_fiber_verification_r1"


def verify(parent, out):
    """Reconstruct all corner actions/statistics without another model call or fit."""
    if out.exists():
        raise FileExistsError("preserve previous verifier output")
    m = check(parent)
    spec = m["spec"]
    old = ROOT / spec["parent"]
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("frozen packet output changed " + name)
    design = read_json(parent / "00/design.json")
    old_design = read_json(old / "00/design.json")
    expected = {
        (p, w, q, i, j)
        for p in range(4)
        for w in range(4)
        for q in spec["queries"]
        for i in range(2)
        for j in range(2)
    }
    if (
        len(design) != 192
        or {
            (r["pair"], r["wording"], r["query"], r["donor_side"], r["receiver_side"])
            for r in design
        }
        != expected
    ):
        raise ValueError("complete48 four-corner units required")
    calls = read_json(parent / "run_status.json")
    if (calls["forward_calls"], calls["backward_calls"], calls["in_flight"]) != (
        480,
        96,
        None,
    ):
        raise ValueError("480 actual new forwards/96 backwards required")
    barrier_path = parent / "01/source_barrier.json"
    barrier = read_json(barrier_path)
    if barrier["new_fits"] != 0 or barrier["parent_source_sha256"] != digest(
        old / "01/source_manifest.json"
    ):
        raise ValueError("checkpoint changed existing fields or added a fit")
    for name, sha in barrier["source_artifacts"].items():
        if digest(parent / name) != sha:
            raise ValueError("source artifact changed after checkpoint")
    errors = {
        k: 0.0
        for k in (
            "native",
            "action",
            "array",
            "metric",
            "decomposition",
            "contrast",
            "hook",
            "same_h",
            "same_delta",
        )
    }

    def compare(kind, actual, reported):
        if actual is None:
            if reported not in (None, ""):
                raise ValueError("undefined statistic fabricated")
        else:
            errors[kind] = max(
                errors[kind],
                float(
                    np.max(abs(np.asarray(actual, float) - np.asarray(reported, float)))
                ),
            )

    def replay(meta):
        compare(
            "native",
            score(meta["native"], m["alphabet"]["values"]),
            meta["native"]["conditional_score"],
        )

    matrix_exact, inverse_exact = rational_inverse()
    matrix, inverse = np.asarray(matrix_exact, float), np.asarray(inverse_exact, float)
    fitted = {}
    for key, entry in read_json(old / "01/source_manifest.json").items():
        for name, sha in {
            entry["file"]: entry["sha256"],
            **entry["source_hashes"],
        }.items():
            if digest(old / name) != sha:
                raise ValueError("frozen source field changed")
        with np.load(old / entry["file"], allow_pickle=False) as data:
            fitted[key] = data["covector"].copy()
    if len(fitted) != 15:
        raise ValueError("all15 unchanged fields required")
    with np.load(parent / "00/directions.npz", allow_pickle=False) as data:
        bank = {k: data[k].copy() for k in data.files}
    directions = {}
    for q in spec["queries"]:
        b = bank[q + "_b"] / np.linalg.norm(bank[q + "_b"])
        directions[q] = {
            a: np.outer(
                inverse @ ([1.0, 0, 0, 0] if a == "known_mean" else [0.0, -1, 0, 1]), b
            )
            for a in spec["arms"]
        }
    report_actions = {
        (int(a["unit"]), a["arm"]): a for a in table(parent / "02/actions.csv")
    }
    report_endpoints = {
        (int(a["unit"]), a["arm"], a["sign"]): a
        for a in table(parent / "02/endpoints.csv")
    }
    refs, actions, displacements = {}, {}, {}
    new_refs, new_ends = 0, 0
    for r in design:
        i, j = r["donor_side"], r["receiver_side"]
        donor, receiver = old_design[r["donor_unit"]], old_design[r["receiver_unit"]]
        if (
            (donor["context"], receiver["context"])
            != (
                spec["context_pairs"][r["pair"]][i],
                spec["context_pairs"][r["pair"]][j],
            )
            or donor["role"] != receiver["role"]
            or r["role"] != donor["role"]
            or r["diagonal"] != (i == j)
        ):
            raise ValueError("role or donor/receiver pairing changed")
        if r["input_ids"] != receiver["input_ids"] or r["spans"] != receiver["spans"]:
            raise ValueError("receiver token layout changed")
        did = [donor["input_ids"][p] for p in donor["spans"]["A_property"]["positions"]]
        rid = [
            receiver["input_ids"][p]
            for p in receiver["spans"]["A_property"]["positions"]
        ]
        if (
            did != rid
            or did != r["property_token_ids"]
            or donor["spans"]["A_property"]["raw"]
            != receiver["spans"]["A_property"]["raw"]
        ):
            raise ValueError("property tokens/bytes differ")
        donor_path = old / "01/baselines" / f"unit_{r['donor_unit']:03d}.json"
        receiver_path = old / "01/baselines" / f"unit_{r['receiver_unit']:03d}.json"
        path = (
            donor_path
            if r["diagonal"]
            else parent / "01/baselines" / f"unit_{r['unit']:03d}.json"
        )
        meta = read_json(path)
        if digest(path.with_suffix(".npz")) != meta["array_sha256"] or meta[
            "address"
        ] != (donor if r["diagonal"] else r):
            raise ValueError("reference address/array hash changed")
        replay(meta)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            h, g, qstate = (
                data["h"].copy(),
                data["gradient"].astype(float),
                data["q"].copy(),
            )
            receiver_saved = None if r["diagonal"] else data["receiver_h"].copy()
        with np.load(donor_path.with_suffix(".npz"), allow_pickle=False) as data:
            compare("same_h", h, data["h"])
        with np.load(receiver_path.with_suffix(".npz"), allow_pickle=False) as data:
            compare("same_h", qstate, data["q"])
            if receiver_saved is not None:
                compare("same_h", receiver_saved, data["h"])
        if not r["diagonal"]:
            new_refs += 1
            if any(meta["audit"].values()):
                raise ValueError("new baseline intervention audit failed")
            t, checkpoint = (
                datetime.fromisoformat(meta["acquired_at"]),
                datetime.fromisoformat(barrier["frozen_at"]),
            )
            if (
                r["role"] == "source"
                and (t >= checkpoint or meta["source_sha256"] is not None)
            ) or (
                r["role"] != "source"
                and (t <= checkpoint or meta["source_sha256"] != digest(barrier_path))
            ):
                raise ValueError("source/held acquisition order violated")
        refs[r["unit"]] = (h, g, meta["native"]["conditional_score"])
        for arm in spec["arms"]:
            direction = directions[r["query"]][arm]
            intended = (
                direction
                / np.linalg.norm(direction)
                * spec["radius_relative_span_norm"]
                * np.linalg.norm(h)
            )
            plus = (h + intended.astype(np.float32)).astype(np.float32)
            minus = (h - intended.astype(np.float32)).astype(np.float32)
            delta = (plus.astype(float) - minus.astype(float)) / 2
            displacements[r["unit"], arm] = delta
            ends = {}
            for sign in ("plus", "minus"):
                ep = (
                    (old if r["diagonal"] else parent)
                    / "01/patches"
                    / f"unit_{(r['donor_unit'] if r['diagonal'] else r['unit']):03d}_{arm}_{sign}.json"
                )
                end = read_json(ep)
                replay(end)
                if (
                    end["address"]
                    != {
                        "unit": r["donor_unit"] if r["diagonal"] else r["unit"],
                        "arm": arm,
                        "sign": sign,
                    }
                    or end["baseline_sha256"] != digest(path)
                    or any(end["audit"].values())
                ):
                    raise ValueError("endpoint address or intervention audit failed")
                if not r["diagonal"]:
                    new_ends += 1
                    et, checkpoint = (
                        datetime.fromisoformat(end["acquired_at"]),
                        datetime.fromisoformat(barrier["frozen_at"]),
                    )
                    if end["source_sha256"] != meta["source_sha256"] or (
                        r["role"] == "source"
                    ) != (et < checkpoint):
                        raise ValueError("endpoint checkpoint chronology changed")
                ends[sign] = end["native"]["conditional_score"]
                compare(
                    "native",
                    ends[sign],
                    report_endpoints[r["unit"], arm, sign]["conditional_score"],
                )
            moments = matrix @ delta
            a = {
                "odd_action": (ends["plus"] - ends["minus"]) / 2,
                "even_change": (ends["plus"] + ends["minus"]) / 2 - refs[r["unit"]][2],
                "plus_change": ends["plus"] - refs[r["unit"]][2],
                "minus_change": ends["minus"] - refs[r["unit"]][2],
                "physical_delta_norm": float(np.sqrt(np.sum(delta * delta))),
                "global_source": float(
                    np.sum(fitted[r["query"] + "_global"] * moments)
                ),
                "wording_source": float(
                    np.sum(fitted[r["query"] + f"_wording{r['wording']}"] * moments)
                ),
                "local_derivative": float(np.sum(g * delta)),
            }
            for k, v in a.items():
                compare("action", v, report_actions[r["unit"], arm][k])
            actions[r["unit"], arm] = {**r, "arm": arm, **a}
    if (new_refs, new_ends, len(report_actions), len(report_endpoints)) != (
        96,
        384,
        384,
        768,
    ):
        raise ValueError("new/inherited actual coverage wrong")
    rm = lambda a: float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))
    metrics = table(parent / "02/metrics.csv")
    if len(metrics) != 36:
        raise ValueError("36cold metrics required")
    for row in metrics:
        items = [
            a
            for a in actions.values()
            if (a["role"], a["query"], a["arm"], a["diagonal"])
            == (row["role"], row["query"], row["arm"], row["diagonal"] == "True")
        ]
        y = np.array([a["odd_action"] for a in items])
        d = np.array([a["physical_delta_norm"] for a in items])
        values = {
            "cells": len(items),
            "contexts": len({a["context"] for a in items}),
            "wordings": len({a["wording"] for a in items}),
            "native_action_rms": rm(y),
            "native_unit_action_rms": rm(y / d),
            "even_change_rms": rm([a["even_change"] for a in items]),
        }
        for method in ("global_source", "wording_source", "local_derivative"):
            prediction = np.array([a[method] for a in items])
            values.update(
                {
                    method + "_rmse": rm(prediction - y),
                    method + "_unit_rmse": rm((prediction - y) / d),
                    method + "_prediction_rms": rm(prediction),
                    method + "_signs": int(sum(prediction * y > 0)),
                }
            )
        for k, v in values.items():
            compare("metric", v, row[k])
    decomp = table(parent / "02/corner_decomposition.csv")
    ctable = table(parent / "02/fiber_contrasts.csv")
    if len(decomp) != 576 or len(ctable) != 1728:
        raise ValueError("all12spaces per48units and contrasts required")
    with np.load(parent / "02/corner_arrays.npz", allow_pickle=False) as arrays:
        for row in decomp:
            p, w, q, space, arm = (
                int(row["pair"]),
                int(row["wording"]),
                row["query"],
                row["space"],
                row["arm"],
            )
            by = {
                (r["donor_side"], r["receiver_side"]): r
                for r in design
                if (r["pair"], r["wording"], r["query"]) == (p, w, q)
            }
            g = np.array(
                [[refs[by[i, j]["unit"]][1] for j in range(2)] for i in range(2)]
            )
            if space == "gradient":
                x = g
            elif space == "covector":
                x = np.einsum("ab,ijbd->ijad", inverse.T, g)
            elif space == "task_gradient":
                x = np.array(
                    [
                        [
                            [
                                float(
                                    np.sum(
                                        g[i, j]
                                        * directions[q][a]
                                        / np.linalg.norm(directions[q][a])
                                    )
                                )
                                for a in spec["arms"]
                            ]
                            for j in range(2)
                        ]
                        for i in range(2)
                    ]
                )
            elif space == "native_level":
                x = np.array(
                    [[refs[by[i, j]["unit"]][2] for j in range(2)] for i in range(2)]
                )
            else:
                method = (
                    "wording_source"
                    if space == "cold_unit_action"
                    else "local_derivative"
                    if space == "local_unit_action"
                    else "odd_action"
                )
                x = np.array(
                    [
                        [
                            actions[by[i, j]["unit"], arm][method]
                            / (
                                1
                                if space == "raw_action"
                                else actions[by[i, j]["unit"], arm][
                                    "physical_delta_norm"
                                ]
                            )
                            for j in range(2)
                        ]
                        for i in range(2)
                    ]
                )
            key = f"p{p}_w{w}_{q}_{space}_{arm}"
            compare("array", x, arrays[key])
            parts, stats = alternate_parts(x)
            rename = {"context": "donor", "wording": "receiver"}
            for k, value in parts.items():
                compare("array", value, arrays[key + "_" + rename.get(k, k)])
            for k, value in stats.items():
                name = next(
                    (k.replace(a, b) for a, b in rename.items() if k.startswith(a)), k
                )
                compare("decomposition", value, row[name])
            for i in range(2):
                for a in spec["arms"]:
                    compare(
                        "same_delta",
                        displacements[by[i, 0]["unit"], a],
                        displacements[by[i, 1]["unit"], a],
                    )
            selected = [
                c
                for c in ctable
                if (int(c["pair"]), int(c["wording"]), c["query"], c["space"], c["arm"])
                == (p, w, q, space, arm)
            ]
            for c in selected:
                side = int(c["fixed_side"])
                a, b = (
                    (x[side, 0], x[side, 1])
                    if c["vary"] == "receiver"
                    else (x[0, side], x[1, side])
                )
                delta = a - b
                diff = float(np.sqrt(np.sum(delta * delta)))
                ref = float(np.sqrt((np.sum(a * a) + np.sum(b * b)) / 2))
                for k, v in {
                    "difference_norm": diff,
                    "reference_rms_norm": ref,
                    "relative_difference": diff / ref if ref else None,
                    "signed_difference": float(delta) if delta.ndim == 0 else None,
                }.items():
                    compare("contrast", v, c[k])
    if any(v > 1e-10 for v in errors.values()):
        raise ValueError("alternate reconstruction mismatch " + str(errors))
    out.mkdir(parents=True)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM31_NOT_GLOBAL",
            "counts": read_json(parent / "02/summary.json")["counts"],
            "maximum_errors": errors,
            "artifact_hashes_checked": len(ledger),
            "inherited_hashes_checked": len(m["files"]),
            "source_barrier_sha256": digest(barrier_path),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "independence": "same-agent alternate score/rational/projection formulas, not third-party or psychological validation",
            "new_model_calls": 0,
            "new_fits": 0,
        },
    )
    print(read_json(out / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.parent, args.out)

#!/usr/bin/env python3
"""HM53 alternate arithmetic and full source/application assignment verification."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_answer_recoding_mechanism import (
    IDENTITY,
    OUT,
    capture_path,
    check,
    write,
)
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.verify_grain_math_evaluative_routing import native_score
from scripts.verify_grain_math_relational_cut import csv_rows
from scripts.verify_grain_math_relational_operation_specificity import compare, odds


def scalar_readout(native, code):
    """Recompute raw moments with scalar sums, then apply the declared affine map."""
    score, mass = native_score(native)
    log = odds(native)
    return {
        "raw_score": score,
        "semantic_score": score if code == 1 else 1 - score,
        "raw_log_odds": log,
        "semantic_log_odds": log if code == 1 else -log,
        "candidate_mass": mass,
    }


def mapping(a, b, allow_actor=False):
    """Check raw ordered token identities independently of the preparation helper."""
    if a["content"] != b["content"] or a["query"] != b["query"]:
        raise ValueError("Target mismatch")
    p, q = a["partition"]["suffix"], b["partition"]["suffix"]
    x, y = [a["input_ids"][i] for i in p], [b["input_ids"][i] for i in q]
    if len(x) != len(y) or a["cue_slots"] != b["cue_slots"]:
        raise ValueError("Suffix length/cue mismatch")
    allowed = set(a["cue_slots"]) if a["codec"] != b["codec"] else set()
    if a["person"] != b["person"]:
        if not allow_actor or a["actor_slot"] != b["actor_slot"]:
            raise ValueError("Undeclared wrong person")
        allowed.add(a["actor_slot"])
    if {i for i in range(len(x)) if x[i] != y[i]} != allowed:
        raise ValueError("Unexpected changed suffix token")
    return {
        "source_positions": p,
        "receiver_positions": q,
        "changed_slots": sorted(allowed),
    }


def verify(out):
    """Reconstruct all48source rows/648calls/576patches/288orbits without neural calls."""
    m, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 648
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["model_seconds"] > 600
    ):
        raise ValueError("Incomplete finite acquisition")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / name) != sha:
            raise ValueError("Changed actual output " + name)
    seal = read_json(out / "01/source_seal.json")
    if (
        not seal["before_application_calls"]
        or not seal["before_application_payload_loading"]
        or seal["completed_new_source_calls"] != 24
        or seal["fit_or_answer_selection"]
    ):
        raise ValueError("Source barrier violated")
    if digest(out / "01/source_vectors.npz") != seal["source_vectors_sha256"]:
        raise ValueError("Source vectors changed")
    for path, sha in seal["source_capture_hashes"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Source provenance changed")
    rows = {r["unit"]: r for r in read_json(out / "00/rows.json")}
    design = read_json(out / "00/design.json")
    if (
        len(rows) != 144
        or len(design) != 648
        or sum(r["reused"] for r in rows.values()) != 72
    ):
        raise ValueError("Missing planned scope")
    index = {
        (
            r["role"],
            r["content"],
            r["class"],
            r["person"],
            r["direction"],
            r["codec"],
        ): r
        for r in rows.values()
    }
    captures, readouts, natives, errors = {}, {}, [], []
    for r in rows.values():
        if r["codec"] == 1:
            other = index[
                r["role"], r["content"], r["class"], r["person"], r["direction"], -1
            ]
            if other["rendered"] != r["rendered"].replace(
                "Use 0 for very low, 9 for very high",
                "Use 9 for very low, 0 for very high",
            ):
                raise ValueError("Text recoding changed another field")
            if len(r["input_ids"]) != len(other["input_ids"]):
                raise ValueError("Full input lengths differ")
            changed = {
                i
                for i in range(len(r["input_ids"]))
                if r["input_ids"][i] != other["input_ids"][i]
            }
            if (
                changed != {r["partition"]["suffix"][s] for s in r["cue_slots"]}
                or mapping(r, other) != other["code_proof"]
            ):
                raise ValueError("Only two actual cue IDs may change")
        meta, a = load_arrays(capture_path(out, r))
        if not r["reused"]:
            e = design[r["capture_unit"]]
            if (
                e["kind"] != "capture"
                or e["receiver"] != r["unit"]
                or any(meta[k] != e[k] for k in IDENTITY)
            ):
                raise ValueError("New capture identity drift")
            if (meta["finished_epoch"] < seal["created_epoch"]) != (
                r["role"] == "source"
            ):
                raise ValueError("Actual source/application time barrier violated")
        elif any(meta[k] != r[k] for k in ("content", "person", "direction")):
            raise ValueError("Reused capture attribution drift")
        if len(a["full"]) != len(r["input_ids"]):
            raise ValueError("Full state/token mismatch")
        v = scalar_readout(meta["native"], r["codec"])
        errors.extend(
            (
                abs(v["raw_score"] - meta["native"]["conditional_score"]),
                abs(v["candidate_mass"] - meta["native"]["candidate_mass"]),
            )
        )
        captures[r["unit"]], readouts[r["unit"]] = a["full"], v
        natives.append(
            {
                k: r[k]
                for k in (
                    "unit",
                    "role",
                    "class",
                    "content",
                    "person",
                    "direction",
                    "codec",
                    "reused",
                )
            }
            | v
            | {"tokens": len(a["full"])}
        )
    source = {}
    with np.load(out / "01/source_vectors.npz", allow_pickle=False) as archive:
        for target in m["spec"]["targets"]:
            for person in ("A", "B"):
                for codec in (1, -1):
                    a, b = [
                        index["source", target, "source", person, d, codec]
                        for d in (0, 1)
                    ]
                    mapping(a, b)
                    delta = np.subtract(
                        captures[b["unit"]][b["partition"]["suffix"]].astype(float),
                        captures[a["unit"]][a["partition"]["suffix"]].astype(float),
                    )
                    if not np.array_equal(
                        delta, archive[f"{target}__{person}__{codec}"]
                    ):
                        raise ValueError(
                            "Source delta not reconstructed from source captures"
                        )
                    source[target, person, codec] = delta
    parent = ROOT / m["spec"]["parent"]
    hm51 = ROOT / read_json(parent / "00/manifest.json")["spec"]["parent"]
    with np.load(hm51 / "00/source_vectors.npz", allow_pickle=False) as archive:
        for target in m["spec"]["targets"]:
            for person in ("A", "B"):
                if not np.array_equal(
                    source[target, person, 1], archive[f"{target}__{person}"]
                ):
                    raise ValueError("Original standard source operator changed")
    effects, max_pre, max_noop, max_random = [], 0.0, 0.0, 0.0
    for e in design[72:]:
        r, d = rows[e["receiver"]], rows[e["donor"]]
        if (
            e["kind"] != "patch"
            or d["direction"] != 1 - r["direction"]
            or d["class"] != r["class"]
            or d["codec"] != r["codec"]
        ):
            raise ValueError("Wrong fixed donor")
        proof = mapping(d, r)
        if proof["source_positions"] != e["donor_positions"]:
            raise ValueError("Donor positions drift")
        src = index[
            "source", r["content"], "source", e["source_actor"], 0, e["source_code"]
        ]
        if (
            mapping(src, r, allow_actor=e["arm"] == "wrong_person")
            != e["source_mapping"]
        ):
            raise ValueError("Source code/person mapping drift")
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        if any(meta[k] != e[k] for k in IDENTITY):
            raise ValueError("Intervention identity drift")
        p, q = r["partition"]["suffix"], e["donor_positions"]
        current = captures[r["unit"]]
        arm = e["arm"]
        if arm == "exact_suffix":
            value = captures[d["unit"]][q]
        elif arm == "noop":
            value = current[p]
        else:
            code = (
                1
                if arm == "source_forward"
                else -1
                if arm == "source_reverse"
                else r["codec"]
            )
            person = (
                ("B" if r["person"] == "A" else "A")
                if arm == "wrong_person"
                else r["person"]
            )
            if (code, person) != (e["source_code"], e["source_actor"]):
                raise ValueError("Source selection differs from predeclared arm")
            delta = source[r["content"], person, code]
            norms = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            if arm == "random":
                noise = np.random.default_rng(
                    m["spec"]["random_seed"] + r["standard_reference_uid"] // 2
                ).normal(size=delta.shape)
                delta = (
                    noise
                    * (norms / np.sqrt(np.einsum("ij,ij->i", noise, noise)))[:, None]
                )
            value = (
                current[p].astype(float) + (1 - 2 * r["direction"]) * delta
            ).astype(current.dtype)
            if arm == "random":
                observed = np.linalg.norm(
                    value.astype(float) - current[p].astype(float), axis=1
                )
                max_random = max(
                    max_random,
                    float(np.max(abs(observed - norms) / np.maximum(norms, 1e-20))),
                )
        # Scalar norm summation can change the final float32 rounding by one ULP.
        difference = float(
            np.max(abs(value.astype(float) - a["patch_values"].astype(float)))
        )
        if difference > 1e-6 or list(a["patch_positions"]) != p:
            raise ValueError("Incorrect actual patch assignment")
        expected = a["before"].copy()
        expected[p] = a["patch_values"]
        if not np.array_equal(expected, a["full"]):
            raise ValueError("Unselected states changed")
        max_pre = max(
            max_pre,
            float(np.max(abs(a["before"].astype(float) - current.astype(float)))),
        )
        n, donor, v = (
            readouts[r["unit"]],
            readouts[d["unit"]],
            scalar_readout(meta["native"], r["codec"]),
        )
        errors.extend(
            (
                abs(v["raw_score"] - meta["native"]["conditional_score"]),
                abs(v["candidate_mass"] - meta["native"]["candidate_mass"]),
            )
        )
        if arm == "noop":
            max_noop = max(max_noop, abs(v["raw_score"] - n["raw_score"]))
        sign = (1 - 2 * r["direction"]) * (-1 if r["person"] == "A" else 1)
        effects.append(
            {k: e[k] for k in IDENTITY}
            | {
                "source_code": e["source_code"],
                "source_actor": e["source_actor"],
                "candidate_mass": v["candidate_mass"],
                **{
                    prefix + "_" + k: vv[k]
                    for prefix, vv in (("native", n), ("donor", donor), ("patched", v))
                    for k in (
                        "raw_score",
                        "semantic_score",
                        "raw_log_odds",
                        "semantic_log_odds",
                    )
                },
                "raw_action": sign * (v["raw_log_odds"] - n["raw_log_odds"]),
                "semantic_action": sign
                * (v["semantic_log_odds"] - n["semantic_log_odds"]),
                "raw_score_action": sign * (v["raw_score"] - n["raw_score"]),
                "semantic_score_action": sign
                * (v["semantic_score"] - n["semantic_score"]),
            }
        )

    def check_table(name, expected):
        got = csv_rows(out / "03" / name)
        if len(got) != len(expected):
            raise ValueError("Missing table rows " + name)
        for x, y in zip(got, expected, strict=True):
            compare(x, y, errors)

    check_table("native.csv", [r for r in natives if r["role"] == "application"])
    check_table("source_native.csv", [r for r in natives if r["role"] == "source"])
    check_table("effects.csv", effects)
    pairs = []
    ni = {
        (
            r["role"],
            r["content"],
            r["class"],
            r["person"],
            r["direction"],
            r["codec"],
        ): r
        for r in natives
    }
    for key, a in ni.items():
        if key[-1] == 1:
            b = ni[key[:-1] + (-1,)]
            pairs.append(
                {k: a[k] for k in ("role", "content", "class", "person", "direction")}
                | {
                    "standard_raw_score": a["raw_score"],
                    "inverse_raw_score": b["raw_score"],
                    "standard_semantic_score": a["semantic_score"],
                    "inverse_semantic_score": b["semantic_score"],
                    "semantic_score_difference": a["semantic_score"]
                    - b["semantic_score"],
                    "raw_complement_error": math.fsum(
                        (a["raw_score"], b["raw_score"], -1)
                    ),
                    "standard_raw_log_odds": a["raw_log_odds"],
                    "inverse_raw_log_odds": b["raw_log_odds"],
                    "semantic_log_odds_difference": math.fsum(
                        (a["semantic_log_odds"], -b["semantic_log_odds"])
                    ),
                }
            )
    check_table("native_code_pairs.csv", pairs)
    ei = {
        (r["content"], r["class"], r["person"], r["direction"], r["arm"], r["codec"]): r
        for r in effects
    }
    orbits = []
    for key, a in ei.items():
        if key[-1] == 1:
            b = ei[key[:-1] + (-1,)]
            orbits.append(
                {k: a[k] for k in ("content", "class", "person", "direction", "arm")}
                | {
                    "standard_raw_action": a["raw_action"],
                    "inverse_raw_action": b["raw_action"],
                    "raw_even": math.fsum((a["raw_action"], b["raw_action"])) / 2,
                    "raw_odd": math.fsum((a["raw_action"], -b["raw_action"])) / 2,
                    "raw_score_even": math.fsum(
                        (a["raw_score_action"], b["raw_score_action"])
                    )
                    / 2,
                    "raw_score_odd": math.fsum(
                        (a["raw_score_action"], -b["raw_score_action"])
                    )
                    / 2,
                }
            )
    check_table("code_orbits.csv", orbits)
    for name, keys in (
        ("class_summary.csv", ("class", "codec", "arm")),
        ("target_summary.csv", ("content", "class", "codec", "arm")),
        ("person_summary.csv", ("person", "class", "codec", "arm")),
    ):
        grouped = defaultdict(list)
        for r in effects:
            grouped[tuple(r[k] for k in keys)].append(r)
        expected = [
            dict(zip(keys, k, strict=True))
            | {
                "rows": len(v),
                **{
                    field + "_mean": math.fsum(r[field] for r in v) / len(v)
                    for field in (
                        "raw_action",
                        "semantic_action",
                        "raw_score_action",
                        "semantic_score_action",
                    )
                },
                "semantic_direction_count": sum(r["semantic_action"] > 0 for r in v),
            }
            for k, v in grouped.items()
        ]
        check_table(name, expected)
    maximum = max(errors)
    if maximum > 1e-10 or max_pre > 1e-6 or max_noop > 1e-6 or max_random > 1e-5:
        raise ValueError("Engineering arithmetic/assignment tolerance failed")
    write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM53_NOT_GLOBAL",
            "new_calls_checked": 648,
            "source_rows_checked": 48,
            "application_rows_checked": 96,
            "effects_checked": 576,
            "native_code_pairs_checked": len(pairs),
            "code_orbits_checked": len(orbits),
            "maximum_scalar_error": maximum,
            "maximum_prepatch_error": max_pre,
            "maximum_noop_error": max_noop,
            "maximum_random_norm_relative_error": max_random,
            "new_neural_calls": 0,
            "same_agent_alternate_arithmetic_not_independent_neural_replication": True,
        },
    )
    print("VERIFIED:48source/96receivers/576assignments/288orbits", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)

#!/usr/bin/env python3
"""HM22 alternate source covectors, scalar accounts and native covariance blocks."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_query_action_transport import DEFAULT, check
from scripts.run_grain_math_response import write_json
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = ROOT / "results/grain_math_response_v1/query_action_transport_verification_r1"


def rows(path):
    """Read exactly the addressed table, not inferred files or filesystem sidecars."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def probability(z):
    """Independent exp/max/sum softmax without the primary observation helper."""
    z = np.asarray(z, float)
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def weights(families):
    """Derive original equal-family measures from actual original address counts."""
    counts = Counter(families)
    return np.array([1 / len(counts) / counts[f] for f in families])


def covariance(a, b, w):
    """Recompute scalar covariance directly, without the primary matrix moments."""
    return float(np.einsum("i,i,i->", w, a - np.dot(w, a), b - np.dot(w, b)))


def verify(packet, out):
    """Validate all source references, action addresses and signed joint accounts."""
    if out.exists():
        raise FileExistsError("preserve earlier HM22 verification")
    manifest = check(packet, source=True)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for n, sha in ledger.items():
        if digest(packet / n) != sha:
            raise ValueError(f"changed primary artifact: {n}")
    out.mkdir(parents=True)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    errors, counts = defaultdict(float), Counter()

    def compare(saved, expected, category):
        """Check addresses exactly and track ratios separately from physical units."""
        for k, v in expected.items():
            actual = saved[k]
            if v is None:
                if actual != "":
                    raise ValueError(f"{category}/{k}: undefined ratio filled")
            elif isinstance(v, (bool, np.bool_)):
                if actual != str(bool(v)):
                    raise ValueError(f"{category}/{k}: boolean changed")
            elif isinstance(v, (float, int, np.number)):
                err = abs(float(actual) - v)
                part = category + ("_ratio" if "ratio" in k else "")
                errors[part] = max(errors[part], float(err))
                if not np.isclose(
                    float(actual),
                    v,
                    atol=spec["arithmetic_absolute_tolerance"],
                    rtol=spec["aggregate_relative_tolerance"],
                ):
                    raise ValueError(f"{category}/{k}: {actual} != {v}")
            elif isinstance(v, (list, dict)):
                if json.loads(actual) != v:
                    raise ValueError(f"{category}/{k}: structured address changed")
            elif actual != str(v):
                raise ValueError(f"{category}/{k}: address changed")
        counts[category] += 1

    try:
        meta = read_json(packet / "00/metadata.json")
        old = ROOT / spec["readout_packet"]
        a = np.array(read_json(packet / "00/alphabet.json")["values"])
        source_rows = rows(packet / "01/source_addresses.csv")
        audits = {r["query"]: r for r in rows(packet / "01/anchor_audit.csv")}
        anchors, source_ledger = {}, {}
        with np.load(packet / "01/anchors.npz") as p:
            for qi, q in enumerate(spec["queries"]):
                selected = [
                    r
                    for r in meta
                    if r["role"] == "source"
                    and r["arm"] == spec["anchor_design"]
                    and r["query"] == q
                ]
                if (
                    len(selected) != spec["anchor_states_per_query"]
                    or len({r["family"] for r in selected}) != spec["source_families"]
                ):
                    raise ValueError("source definition changed")
                w = weights([r["family"] for r in selected])
                gs = []
                multiplicity = Counter(r["reference_path"] for r in selected)
                by_row = {int(r["row"]): r for r in source_rows if r["query"] == q}
                for i, r in enumerate(selected):
                    source_ledger[r["reference_path"]] = digest(
                        ROOT / r["reference_path"]
                    )
                    pr = probability(
                        read_json(ROOT / r["reference_path"])["native"][
                            "candidate_logits"
                        ]
                    )
                    gs.append(pr * (a - np.dot(pr, a)))
                    compare(
                        by_row[r["row"]],
                        {
                            **r,
                            "anchor_weight": w[i],
                            "reference_multiplicity": multiplicity[r["reference_path"]],
                        },
                        "source_addresses",
                    )
                gs = np.array(gs)
                bar = np.einsum("i,ij->j", w, gs)
                anchors[q] = bar
                for name, expected in (
                    ("rows", np.array([r["row"] for r in selected])),
                    ("weights", w),
                    ("gradients", gs),
                    ("anchor", bar),
                ):
                    err = float(np.max(abs(p[f"q{qi}_{name}"] - expected)))
                    errors["source_arrays"] = max(errors["source_arrays"], err)
                    if err > spec["arithmetic_absolute_tolerance"]:
                        raise ValueError(f"changed source anchor {q}/{name}")
                compare(
                    audits[q],
                    {
                        "query": q,
                        "design": spec["anchor_design"],
                        "states": len(selected),
                        "families": len({r["family"] for r in selected}),
                        "distinct_reference_records": len(multiplicity),
                        "duplicate_address_occurrences": len(selected)
                        - len(multiplicity),
                        "max_reference_multiplicity": max(multiplicity.values()),
                        "anchor_sum_gauge_error": abs(float(bar.sum())),
                        "anchor_norm": float(np.sqrt(np.dot(bar, bar))),
                    },
                    "anchors",
                )
        if source_ledger != read_json(packet / "01/consumed_source_hashes.json"):
            raise ValueError("source record ledger changed")
        actual = rows(packet / "02/actions.csv")
        if len(actual) != spec["paired_query_rows"]:
            raise ValueError("incomplete native actions")
        values, ds = {}, {}
        with (
            np.load(old / "02/endpoints.npz") as previous,
            np.load(packet / "02/accounts.npz") as current,
        ):
            for r, saved in zip(meta, actual, strict=True):
                i, q = r["row"], r["query"]
                zr, zt = previous["reference_logits"][i], previous["target_logits"][i]
                pr, pt = probability(zr), probability(zt)
                d = (zt - np.sum(zt) / len(zt)) - (zr - np.sum(zr) / len(zr))
                g = pr * (a - np.dot(pr, a))
                y, l = float(np.dot(pt - pr, a)), float(np.dot(g, d))
                b = float(np.dot(anchors[q], d))
                v = np.array([b, l - b, y - l])
                err = float(np.max(abs(v - current["parts"][i])))
                errors["action_arrays"] = max(errors["action_arrays"], err)
                if r["exact_input_noop"] and (np.any(v) or np.any(d)):
                    raise ValueError("changed native exact no-op")
                compare(
                    saved,
                    {**r, "B": v[0], "G": v[1], "C": v[2], "native": y, "tangent": l},
                    "actions",
                )
                values[r["state_index"], q] = {**r, "parts": v, "native": y}
                ds[r["state_index"], q] = d
        compiled = {
            (int(r["row"]), r["model"]): r
            for r in rows(ROOT / spec["exchange_packet"] / "02/components.csv")
        }
        oldjoint = {
            (r["role"], r["design"], r["query_a"], r["query_b"]): r
            for r in rows(old / "02/readout_covariance.csv")
        }
        statistics = {
            (r["role"], r["design"], r["query"]): r
            for r in rows(packet / "02/action_statistics.csv")
        }
        joint = {
            (r["role"], r["design"], r["query_a"], r["query_b"]): r
            for r in rows(packet / "02/covariance_accounts.csv")
        }
        gaps = {
            (
                r["role"],
                r["design"],
                r["query_a"],
                r["query_b"],
                r["model"],
                r["channel"],
            ): r
            for r in rows(packet / "02/compiled_gap_decomposition.csv")
        }
        matrix_addresses = rows(packet / "02/logit_matrix_addresses.csv")
        first = [r for r in meta if r["query"] == spec["queries"][0]]
        if sorted(r["state_index"] for r in first) != list(
            range(spec["unique_state_pairs"])
        ):
            raise ValueError("incomplete paired states")
        with np.load(packet / "02/logit_cross_covariance.npz") as matrices:
            matrix_i = 0
            for role, design in itertools.product(
                spec["roles"], ("all_states", *sorted({r["arm"] for r in first}))
            ):
                selected = [
                    r
                    for r in first
                    if r["role"] == role
                    and (design == "all_states" or r["arm"] == design)
                ]
                ids = [r["state_index"] for r in selected]
                w = weights([r["family"] for r in selected])
                info = {
                    "role": role,
                    "design": design,
                    "states": len(ids),
                    "families": len({r["family"] for r in selected}),
                }
                for q in spec["queries"]:
                    v = np.array([values[s, q]["parts"] for s in ids])
                    y = np.array([values[s, q]["native"] for s in ids])
                    rms = lambda x, w=w: float(np.sqrt(np.dot(w, x * x)))
                    compare(
                        statistics[role, design, q],
                        {
                            **info,
                            "query": q,
                            "native_rms": rms(y),
                            **{n + "_rms": rms(v[:, i]) for i, n in enumerate("BGC")},
                            **{
                                n + "_native_rms_ratio": rms(v[:, i]) / rms(y)
                                if rms(y)
                                else None
                                for i, n in enumerate("BGC")
                            },
                            "B_G_inner": float(w @ (v[:, 0] * v[:, 1])),
                            "B_C_inner": float(w @ (v[:, 0] * v[:, 2])),
                            "G_C_inner": float(w @ (v[:, 1] * v[:, 2])),
                        },
                        "statistics",
                    )
                for qa, qb in itertools.combinations(spec["queries"], 2):
                    va, vb = (
                        np.array([values[s, q]["parts"] for s in ids]) for q in (qa, qb)
                    )
                    ya, yb = (
                        np.array([values[s, q]["native"] for s in ids])
                        for q in (qa, qb)
                    )
                    nc = covariance(ya, yb, w)
                    terms = {
                        pa + pb: covariance(va[:, i], vb[:, j], w)
                        for i, pa in enumerate("BGC")
                        for j, pb in enumerate("BGC")
                    }
                    tangent = covariance(
                        va[:, :2].sum(axis=1), vb[:, :2].sum(axis=1), w
                    )
                    sensitivity = terms["BG"] + terms["GB"] + terms["GG"]
                    link = sum(terms[n] for n in ("BC", "GC", "CB", "CG", "CC"))
                    da, db = (np.array([ds[s, q] for s in ids]) for q in (qa, qb))
                    ma, mb = np.einsum("i,ij->j", w, da), np.einsum("i,ij->j", w, db)
                    sigma = np.einsum("i,ij,ik->jk", w, da - ma, db - mb)
                    err = float(np.max(abs(sigma - matrices["matrices"][matrix_i])))
                    errors["logit_blocks"] = max(errors["logit_blocks"], err)
                    if err > spec["arithmetic_absolute_tolerance"]:
                        raise ValueError("native logit covariance block differs")
                    key = {**info, "query_a": qa, "query_b": qb, "matrix_row": matrix_i}
                    compare(matrix_addresses[matrix_i], key, "matrix_addresses")
                    contraction = float(
                        np.einsum("j,jk,k->", anchors[qa], sigma, anchors[qb])
                    )
                    errors["contraction"] = max(
                        errors["contraction"], abs(contraction - terms["BB"])
                    )
                    compare(
                        joint[role, design, qa, qb],
                        {
                            **key,
                            **terms,
                            "native_covariance": nc,
                            "fixed_covariance": terms["BB"],
                            "tangent_covariance": tangent,
                            "sensitivity_increment": sensitivity,
                            "link_increment": link,
                            "recombination_error": abs(nc - tangent - link),
                            "fixed_native_sign_agree": bool(
                                np.sign(nc) == np.sign(terms["BB"])
                            ),
                            "sensitivity_native_ratio": sensitivity / nc
                            if nc
                            else None,
                            "link_native_ratio": link / nc if nc else None,
                            "fixed_covariance_from_logit_block": contraction,
                            "logit_gauge_left_error": float(
                                np.max(abs(np.ones(len(a)) @ sigma))
                            ),
                            "logit_gauge_right_error": float(
                                np.max(abs(sigma @ np.ones(len(a))))
                            ),
                        },
                        "covariance",
                    )
                    for field, expected in (
                        ("native_covariance", nc),
                        ("tangent_covariance", tangent),
                        ("link_covariance_increment", link),
                    ):
                        errors["historical_joint"] = max(
                            errors["historical_joint"],
                            abs(
                                float(oldjoint[role, design, qa, qb][field]) - expected
                            ),
                        )
                    for model, channel in itertools.product(
                        spec["models"], ("forward", "odd")
                    ):
                        ca, cb = (
                            np.array(
                                [
                                    float(compiled[values[s, q]["row"], model][channel])
                                    for s in ids
                                ]
                            )
                            for q in (qa, qb)
                        )
                        pc = covariance(ca, cb, w)
                        gap = nc - pc
                        compare(
                            gaps[role, design, qa, qb, model, channel],
                            {
                                **info,
                                "query_a": qa,
                                "query_b": qb,
                                "model": model,
                                "channel": channel,
                                "native_covariance": nc,
                                "compiled_covariance": pc,
                                "total_gap": gap,
                                "fixed_gap": terms["BB"] - pc,
                                "sensitivity_increment": sensitivity,
                                "link_increment": link,
                                "sensitivity_gap_ratio": sensitivity / gap
                                if gap
                                else None,
                                "link_gap_ratio": link / gap if gap else None,
                                "gap_recombination_error": abs(
                                    nc - terms["BB"] - sensitivity - link
                                ),
                            },
                            "gap",
                        )
                    matrix_i += 1
        expected_counts = {
            "source_addresses": 192,
            "anchors": 3,
            "actions": 2172,
            "statistics": 72,
            "matrix_addresses": 72,
            "covariance": 72,
            "gap": 432,
        }
        if dict(counts) != expected_counts:
            raise ValueError(f"incomplete acceptance coverage: {dict(counts)}")
        if any(
            errors[k] > spec["arithmetic_absolute_tolerance"]
            for k in (
                "action_arrays",
                "logit_blocks",
                "contraction",
                "historical_joint",
            )
        ):
            raise ValueError(f"alternate primitive errors: {dict(errors)}")
        write_json(
            out / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM22_NOT_GLOBAL_SCOPE_COMPLETE",
                "primary_manifest_sha256": digest(packet / "00/manifest.json"),
                "verifier_sha256": digest(Path(__file__)),
                "checked_counts": dict(counts),
                "max_errors": dict(errors),
                "source_reference_files": len(source_ledger),
                "new_source_fits": 0,
                "source_anchor_moments": 3,
                "new_model_calls": 0,
                "same_agent_alternate_arithmetic_not_independent_human_replication": True,
            },
        )
        print(read_json(out / "summary.json"), flush=True)
    except Exception as exc:
        write_json(
            out / "failure_attempt_01.json",
            {
                "status": "FAILED_PRESERVED",
                "error": repr(exc),
                "checked_counts": dict(counts),
                "max_errors": dict(errors),
                "verifier_sha256": digest(Path(__file__)),
            },
        )
        raise


def main():
    """Run one finite verification without simultaneous writers or overwrite."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=DEFAULT)
    p.add_argument("--out", type=Path, default=OUTPUT)
    a = p.parse_args()
    with collector_guard(a.out):
        verify(a.packet, a.out)


if __name__ == "__main__":
    main()

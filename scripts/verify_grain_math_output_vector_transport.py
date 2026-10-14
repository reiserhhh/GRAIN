#!/usr/bin/env python3
"""HM23 alternate kernel, Cholesky solve, readout and complete-table arithmetic."""

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
from scripts.run_grain_math_output_vector_transport import DEFAULT, check
from scripts.run_grain_math_response import write_json
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = ROOT / "results/grain_math_response_v1/output_vector_transport_verification_r1"


def rows(path):
    """Read an explicit addressed table, excluding external metadata files."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def probability(z):
    """Evaluate stable softmax independently of the primary readout helper."""
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def weights(families):
    """Derive equal-family probabilities from actual address multiplicities."""
    c = Counter(families)
    return np.array([1.0 / len(c) / c[f] for f in families])


def covariance(a, b, w):
    """Compute a scalar central moment without the primary moment routine."""
    return float(np.einsum("i,i,i->", w, a - w @ a, b - w @ b))


def metrics(y, p, w):
    """Recompute physical errors and leave constant-vector correlation missing."""
    vy, vp = covariance(y, y, w), covariance(p, p, w)
    mse = float(np.dot(w, (p - y) ** 2))
    return {
        "native_rms": float(np.sqrt(np.dot(w, y * y))),
        "predicted_rms": float(np.sqrt(np.dot(w, p * p))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.dot(w, abs(p - y))),
        "pearson": covariance(y, p, w) / np.sqrt(vy * vp) if vy * vp > 0 else None,
    }


def accounts(pa, pb, w):
    """Rebuild all nine signed B/G/C terms and their original groupings."""
    terms = {
        x + y: covariance(pa[:, i], pb[:, j], w)
        for i, x in enumerate(("B", "G", "C"))
        for j, y in enumerate(("B", "G", "C"))
    }
    native = covariance(pa.sum(axis=1), pb.sum(axis=1), w)
    tangent = covariance(pa[:, :2].sum(axis=1), pb[:, :2].sum(axis=1), w)
    sensitivity = sum(terms[n] for n in ("BG", "GB", "GG"))
    link = sum(terms[n] for n in ("BC", "GC", "CB", "CG", "CC"))
    return {
        **terms,
        "native_covariance": native,
        "tangent_covariance": tangent,
        "fixed_covariance": terms["BB"],
        "sensitivity_increment": sensitivity,
        "link_increment": link,
        "sensitivity_native_ratio": sensitivity / native if native else None,
        "link_native_ratio": link / native if native else None,
        "fixed_native_sign_agree": bool(np.sign(terms["BB"]) == np.sign(native)),
        "recombination_error": abs(native - sum(terms.values())),
    }


def verify(packet, out):
    """Verify source-only construction and every reported held numerical object."""
    if out.exists():
        raise FileExistsError("preserve prior HM23 verification")
    manifest = check(packet, models=True)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError(f"changed primary artifact: {name}")
    out.mkdir(parents=True)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    errors, counts = defaultdict(float), Counter()

    def compare(saved, expected, category):
        """Compare units and address fields; ratios get their own error ledger."""
        for key, value in expected.items():
            actual = saved[key]
            if value is None:
                if actual != "":
                    raise ValueError(f"{category}/{key}: missing value filled")
            elif isinstance(value, (bool, np.bool_)):
                if actual != str(bool(value)):
                    raise ValueError(f"{category}/{key}: boolean changed")
            elif isinstance(value, (float, int, np.number)):
                err = abs(float(actual) - value)
                kind = category + (
                    "_ratio"
                    if "ratio" in key or "pearson" in key or "relative" in key
                    else ""
                )
                errors[kind] = max(errors[kind], float(err))
                if not np.isclose(
                    float(actual),
                    value,
                    atol=spec["arithmetic_absolute_tolerance"],
                    rtol=spec["aggregate_relative_tolerance"],
                ):
                    raise ValueError(f"{category}/{key}: {actual} != {value}")
            elif isinstance(value, (list, dict)):
                if json.loads(actual) != value:
                    raise ValueError(f"{category}/{key}: structured address changed")
            elif actual != str(value):
                raise ValueError(f"{category}/{key}: address changed")
        counts[category] += 1

    def array(actual, expected, category):
        """Require matching shapes and report physical array discrepancies."""
        if actual.shape != expected.shape:
            raise ValueError(f"{category}: shape mismatch")
        err = float(np.max(abs(actual - expected)))
        errors[category] = max(errors[category], err)
        if not np.allclose(
            actual,
            expected,
            atol=spec["arithmetic_absolute_tolerance"],
            rtol=spec["aggregate_relative_tolerance"],
        ):
            raise ValueError(f"{category}: {err}")

    try:
        meta = read_json(packet / "00/metadata.json")
        old = ROOT / spec["context_packet"]
        with np.load(old / "01/source_coordinates.npz") as p:
            origin, sc, sx, states = (
                p[n].copy()
                for n in (
                    "origin",
                    "context_scale",
                    "delta_scale",
                    "source_state_indices",
                )
            )
        with np.load(old / "00/endpoints.npz") as p:
            x, c = p["delta"] / sx, (p["reference"] - origin) / sc
        kernel = np.einsum("ij,kj->ik", x, x[states]) * (
            1 + np.einsum("ij,kj->ik", c, c[states])
        )
        first = {r["state_index"]: r for r in meta if r["query"] == spec["queries"][0]}
        w = weights([first[s]["family"] for s in states])
        function = read_json(packet / "01/function.json")
        unit = function["unit"]
        if not function["source_only"] or function["held_targets_loaded"]:
            raise ValueError("source-only fit declaration contradicted")
        source_rows = rows(packet / "01/source_addresses.csv")
        selected = [r for r in meta if r["role"] == "source"]
        for saved, r in zip(source_rows, selected, strict=True):
            compare(saved, r, "source_addresses")
        if len(selected) != spec["role_rows"]["source"]:
            raise ValueError("source scope changed")
        lookup = {(r["state_index"], r["query"]): r for r in selected}
        source_files, d = {}, []
        for q in spec["queries"]:
            block = []
            for s in states:
                r = lookup[s, q]
                pair = []
                for n in ("reference_path", "unit_path"):
                    source_files[r[n]] = digest(ROOT / r[n])
                    pair.append(
                        np.array(read_json(ROOT / r[n])["native"]["candidate_logits"])
                    )
                block.append((pair[1] - pair[1].mean()) - (pair[0] - pair[0].mean()))
            d.append(block)
        d = np.concatenate(d, axis=1)
        if source_files != read_json(packet / "01/source_files.json"):
            raise ValueError("source endpoint ledger changed")
        root = np.sqrt(w)
        ks = (kernel[states] + kernel[states].T) / (2 * unit)
        system = np.einsum("i,ij,j->ij", root, ks, root) + spec["alpha"] * np.eye(
            len(states)
        )
        chol = np.linalg.cholesky(system)
        dual = root[:, None] * np.linalg.solve(
            chol.T, np.linalg.solve(chol, root[:, None] * d)
        )
        with np.load(packet / "01/model.npz") as p:
            for n, expected in (
                ("source_targets", d),
                ("weights", w),
                ("states", states),
                ("weighted_dual", dual),
            ):
                array(p[n], expected, "source_" + n)
        prediction = np.einsum("ij,jk->ik", kernel, dual) / unit
        alphabet = np.array(read_json(packet / "00/alphabet.json")["values"])
        with np.load(ROOT / spec["query_packet"] / "01/anchors.npz") as p:
            anchors = {
                q: p[f"q{i}_anchor"].copy() for i, q in enumerate(spec["queries"])
            }
        n = len(meta)
        y, new, scalar = np.empty(n), np.empty(n), np.empty(n)
        part, truepart, motion, true_motion = (
            np.empty((n, 10 if i >= 2 else 3)) for i in range(4)
        )
        functions = read_json(old / "01/functions.json")
        for qi, q in enumerate(spec["queries"]):
            f = functions[q]["receiver_tensor"]
            if f["kernel_unit"] != unit:
                raise ValueError("old and new kernel units differ")
            with np.load(old / f["path"]) as p:
                scalar_state = kernel @ p["weighted_dual"] / unit
            for r in (v for v in meta if v["query"] == q):
                i, s = r["row"], r["state_index"]
                zr, zt = (
                    np.array(read_json(ROOT / r[nn])["native"]["candidate_logits"])
                    for nn in ("reference_path", "unit_path")
                )
                pr, pt = probability(zr), probability(zt)
                delta = prediction[s, qi * 10 : (qi + 1) * 10]
                actual_d = (zt - zt.mean()) - (zr - zr.mean())
                y[i] = float(alphabet @ (pt - pr))
                new[i] = float(alphabet @ (probability(zr + delta) - pr))
                scalar[i] = scalar_state[s]
                g = pr * (alphabet - alphabet @ pr)
                b, l = float(anchors[q] @ delta), float(g @ delta)
                bn, ln = float(anchors[q] @ actual_d), float(g @ actual_d)
                part[i], truepart[i] = [b, l - b, new[i] - l], [bn, ln - bn, y[i] - ln]
                motion[i], true_motion[i] = delta, actual_d
                if r["exact_input_noop"] and (np.any(delta) or new[i] or scalar[i]):
                    raise ValueError("nonzero exact no-op")
        with np.load(packet / "02/predictions.npz") as p:
            for name, expected in (
                ("source_order_prediction", prediction),
                ("predicted_delta", motion),
                ("native_delta", true_motion),
                ("predicted_parts", part),
                ("native_parts", truepart),
                ("native", y),
                ("predicted", new),
                ("scalar_receiver", scalar),
            ):
                array(p[name], expected, "application_" + name)
        for saved, r in zip(rows(packet / "02/actions.csv"), meta, strict=True):
            i = r["row"]
            compare(
                saved,
                {
                    **r,
                    "native": y[i],
                    "vector_readout": new[i],
                    "scalar_receiver": scalar[i],
                    "scalar_remainder": y[i] - new[i],
                    "logit_motion_norm": np.sqrt(true_motion[i] @ true_motion[i]),
                    "predicted_motion_norm": np.sqrt(motion[i] @ motion[i]),
                    "motion_error_norm": np.linalg.norm(true_motion[i] - motion[i]),
                    **{
                        f"predicted_{nn}": part[i, j]
                        for j, nn in enumerate(("B", "G", "C"))
                    },
                    **{
                        f"native_{nn}": truepart[i, j]
                        for j, nn in enumerate(("B", "G", "C"))
                    },
                },
                "actions",
            )
        tables = {
            name: rows(packet / "02" / f"{name}.csv")
            for name in (
                "action_statistics",
                "family_comparisons",
                "joint_covariance",
                "account_covariance",
                "matrix_addresses",
            )
        }
        indexed = {
            name: {
                (
                    r["role"],
                    r["design"],
                    r.get("query", r.get("query_a")),
                    r.get("query_b", r.get("family", "")),
                ): r
                for r in values
            }
            for name, values in tables.items()
        }
        row_lookup = {(r["state_index"], r["query"]): r["row"] for r in meta}
        block_index = 0
        with np.load(packet / "02/remainder_covariance_blocks.npz") as saved_blocks:
            for role, design in itertools.product(
                spec["roles"],
                ("all_states", *sorted({r["arm"] for r in first.values()})),
            ):
                selected = [
                    r
                    for _, r in sorted(first.items())
                    if r["role"] == role
                    and (design == "all_states" or r["arm"] == design)
                ]
                ss, fs = (
                    [r["state_index"] for r in selected],
                    [r["family"] for r in selected],
                )
                ww = weights(fs)
                info = {
                    "role": role,
                    "design": design,
                    "states": len(ss),
                    "families": len(set(fs)),
                }
                for q in spec["queries"]:
                    ids = np.array([row_lookup[s, q] for s in ss])
                    family_values = []
                    for ff in sorted(set(fs)):
                        mask = np.array([f == ff for f in fs])
                        value = float(
                            (
                                (new[ids][mask] - y[ids][mask]) ** 2
                                - (scalar[ids][mask] - y[ids][mask]) ** 2
                            ).mean()
                        )
                        family_values.append(value)
                        compare(
                            indexed["family_comparisons"][role, design, q, str(ff)],
                            {
                                **info,
                                "query": q,
                                "family": ff,
                                "observations": int(mask.sum()),
                                "mse_delta_vector_minus_scalar": value,
                            },
                            "families",
                        )
                    fv = np.array(family_values)
                    rng = np.random.default_rng(spec["bootstrap_seed"])
                    indices = rng.integers(
                        0, len(fv), (spec["bootstrap_replicates"], len(fv))
                    )
                    ci = np.quantile(np.take(fv, indices).mean(axis=1), [0.025, 0.975])
                    vm, sm = (
                        metrics(y[ids], new[ids], ww),
                        metrics(y[ids], scalar[ids], ww),
                    )
                    mse_motion = float(
                        np.einsum(
                            "i,ij,ij->",
                            ww,
                            motion[ids] - true_motion[ids],
                            motion[ids] - true_motion[ids],
                        )
                    )
                    norm2 = float(
                        np.einsum("i,ij,ij->", ww, true_motion[ids], true_motion[ids])
                    )
                    compare(
                        indexed["action_statistics"][role, design, q, ""],
                        {
                            **info,
                            "query": q,
                            **{f"vector_{k}": v for k, v in vm.items()},
                            **{f"scalar_{k}": v for k, v in sm.items()},
                            "motion_rmse": np.sqrt(mse_motion),
                            "motion_relative_rmse": np.sqrt(mse_motion / norm2)
                            if norm2
                            else None,
                            "mse_delta_mean": fv.mean(),
                            "mse_delta_ci_low": ci[0],
                            "mse_delta_ci_high": ci[1],
                            "positive_gain_families": sum(v < 0 for v in fv),
                        },
                        "statistics",
                    )
                for qa, qb in itertools.combinations(spec["queries"], 2):
                    ia, ib = (
                        np.array([row_lookup[s, q] for s in ss]) for q in (qa, qb)
                    )
                    key = {**info, "query_a": qa, "query_b": qb}
                    pa, pb, ra, rb = new[ia], new[ib], y[ia] - new[ia], y[ib] - new[ib]
                    pp, pr, rp, rr = (
                        covariance(a, b, ww)
                        for a, b in ((pa, pb), (pa, rb), (ra, pb), (ra, rb))
                    )
                    native_cov, old_cov = (
                        covariance(y[ia], y[ib], ww),
                        covariance(scalar[ia], scalar[ib], ww),
                    )
                    compare(
                        indexed["joint_covariance"][role, design, qa, qb],
                        {
                            **key,
                            "PP": pp,
                            "PR": pr,
                            "RP": rp,
                            "RR": rr,
                            "native_covariance": native_cov,
                            "recombination_error": abs(native_cov - pp - pr - rp - rr),
                            "scalar_covariance": old_cov,
                            "vector_covariance_gap": native_cov - pp,
                            "scalar_covariance_gap": native_cov - old_cov,
                            "vector_native_sign_agree": np.sign(pp)
                            == np.sign(native_cov),
                            "scalar_native_sign_agree": np.sign(old_cov)
                            == np.sign(native_cov),
                        },
                        "joint",
                    )
                    covs = []
                    for a, b in (
                        (motion[ia], motion[ib]),
                        (motion[ia], true_motion[ib] - motion[ib]),
                        (true_motion[ia] - motion[ia], motion[ib]),
                        (true_motion[ia] - motion[ia], true_motion[ib] - motion[ib]),
                    ):
                        covs.append(
                            np.einsum("i,ij,ik->jk", ww, a - ww @ a, b - ww @ b)
                        )
                    array(
                        saved_blocks["blocks"][block_index],
                        np.array(covs),
                        "matrix_blocks",
                    )
                    compare(
                        indexed["matrix_addresses"][role, design, qa, qb],
                        {**key, "matrix_row": block_index},
                        "matrix_addresses",
                    )
                    nv, tv = (
                        accounts(part[ia], part[ib], ww),
                        accounts(truepart[ia], truepart[ib], ww),
                    )
                    compare(
                        indexed["account_covariance"][role, design, qa, qb],
                        {
                            **key,
                            **{f"predicted_{k}": v for k, v in nv.items()},
                            **{f"native_{k}": v for k, v in tv.items()},
                            **{
                                f"motion_{nn}_fixed_contraction": np.einsum(
                                    "i,ij,j->", anchors[qa], covs[j], anchors[qb]
                                )
                                for j, nn in enumerate(("PP", "PR", "RP", "RR"))
                            },
                        },
                        "accounts",
                    )
                    block_index += 1
        if counts != Counter(
            {
                "source_addresses": 1080,
                "actions": 2172,
                "families": spec["family_comparison_rows"],
                "statistics": 72,
                "joint": 72,
                "accounts": 72,
                "matrix_addresses": 72,
            }
        ):
            raise ValueError(f"incomplete verification: {counts}")
        if any(
            len(values)
            != counts[
                {
                    "action_statistics": "statistics",
                    "family_comparisons": "families",
                    "joint_covariance": "joint",
                    "account_covariance": "accounts",
                    "matrix_addresses": "matrix_addresses",
                }[name]
            ]
            for name, values in tables.items()
        ):
            raise ValueError("unverified duplicate/missing table rows")
        write_json(
            out / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM23_NOT_GLOBAL_SCOPE_COMPLETE",
                "primary_manifest_sha256": digest(packet / "00/manifest.json"),
                "verifier_sha256": digest(Path(__file__)),
                "checked_counts": dict(counts),
                "source_shape": list(d.shape),
                "source_files_checked": len(source_files),
                "max_errors": dict(errors),
                "alternate_factorization": "Cholesky two triangular solves",
                "new_source_fits": 0,
                "new_model_calls": 0,
                "held_reference_is_observer_cache": True,
                "same_agent_alternate_arithmetic_not_independent_human_replication": True,
            },
        )
        print(read_json(out / "summary.json"), flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json",
            {
                "status": "FAILED",
                "error": repr(error),
                "counts": dict(counts),
                "errors": dict(errors),
            },
        )
        raise


def main():
    """Run a finite alternate check without altering any primary output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    collector_guard(ROOT)
    verify(args.packet, args.out)


if __name__ == "__main__":
    main()

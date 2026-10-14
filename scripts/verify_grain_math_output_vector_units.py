#!/usr/bin/env python3
"""HM23 companion: physical-unit replay and conditioned ratio error bounds.

The original point-ratio failure remains a failure. This companion neither
changes its tolerances nor claims that an ill-conditioned ratio is precise.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_output_vector_transport import DEFAULT, check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard
from scripts.verify_grain_math_output_vector_transport import (
    accounts,
    covariance,
    metrics,
    probability,
    rows,
    weights,
)

OUT = ROOT / "results/grain_math_response_v1/output_vector_units_verification_r1"
FIRST = ROOT / "results/grain_math_response_v1/output_vector_transport_verification_r1"


def covariance_bound(a, b, ea, eb, w):
    """Bound covariance change by weighted centered-norm contraction.

    Here ea/eb bound the sup norm of the two vector changes. The additional
    gamma term covers the finite dot-product arithmetic, not an empirical fit.
    """
    sa = np.sqrt(max(0.0, covariance(a, a, w)))
    sb = np.sqrt(max(0.0, covariance(b, b, w)))
    gamma = 64 * len(w) * np.finfo(float).eps
    return float(ea * sb + eb * sa + ea * eb + gamma * (sa + ea) * (sb + eb))


def ratio_bound(n, d, en, ed):
    """Bound a quotient perturbation only when its denominator excludes zero."""
    if abs(d) <= ed:
        return None
    return float((en + abs(n / d) * ed) / (abs(d) - ed))


def verify(packet, out):
    """Reconstruct the shared solve and replay all physical tables, retaining failure."""
    if out.exists():
        raise FileExistsError("preserve companion output")
    manifest = check(packet, models=True)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError(f"changed primary: {name}")
    first = read_json(FIRST / "failure.json")
    if first["status"] != "FAILED":
        raise ValueError("original ratio failure must be retained")
    out.mkdir(parents=True)
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    write_json(out / "original_failure.json", first)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    errors, counts, ratios = {}, Counter(), []

    def physical(actual, expected, label):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid array {label}")
        err = float(np.max(abs(a - b), initial=0))
        errors[label] = max(errors.get(label, 0.0), err)
        if not np.allclose(
            a,
            b,
            atol=spec["arithmetic_absolute_tolerance"],
            rtol=spec["aggregate_relative_tolerance"],
        ):
            raise ValueError(f"physical-unit failure {label}: {err}")

    def table(saved, expected, label):
        for k, v in expected.items():
            if "ratio" in k:
                continue
            if v is None:
                if saved[k] != "":
                    raise ValueError(f"missing filled {label}/{k}")
            elif isinstance(v, (bool, np.bool_)):
                if saved[k] != str(bool(v)):
                    raise ValueError(f"boolean {label}/{k}")
            elif isinstance(v, (int, float, np.number)):
                physical(float(saved[k]), v, label)
            elif saved[k] != str(v):
                raise ValueError(f"address {label}/{k}")
        counts[label] += 1

    try:
        meta = read_json(packet / "00/metadata.json")
        context = ROOT / spec["context_packet"]
        with np.load(context / "01/source_coordinates.npz") as p:
            origin, sc, sx, states = (
                p[n].copy()
                for n in (
                    "origin",
                    "context_scale",
                    "delta_scale",
                    "source_state_indices",
                )
            )
        with np.load(context / "00/endpoints.npz") as p:
            x, c = p["delta"] / sx, (p["reference"] - origin) / sc
        k = np.einsum("ij,kj->ik", x, x[states]) * (
            1 + np.einsum("ij,kj->ik", c, c[states])
        )
        first_rows = {
            r["state_index"]: r for r in meta if r["query"] == spec["queries"][0]
        }
        w = weights([first_rows[s]["family"] for s in states])
        unit = read_json(packet / "01/function.json")["unit"]
        source = {
            (r["state_index"], r["query"]): r for r in meta if r["role"] == "source"
        }
        d = []
        for q in spec["queries"]:
            block = []
            for s in states:
                r = source[s, q]
                zr, zt = (
                    np.array(read_json(ROOT / r[n])["native"]["candidate_logits"])
                    for n in ("reference_path", "unit_path")
                )
                block.append((zt - zt.mean()) - (zr - zr.mean()))
            d.append(block)
        d = np.concatenate(d, axis=1)
        sw = np.sqrt(w)
        system = sw[:, None] * ((k[states] + k[states].T) / (2 * unit)) * sw[None, :]
        system += spec["alpha"] * np.eye(len(states))
        chol = np.linalg.cholesky(system)
        dual = sw[:, None] * np.linalg.solve(
            chol.T, np.linalg.solve(chol, sw[:, None] * d)
        )
        with np.load(packet / "01/model.npz") as p:
            physical(p["weighted_dual"], dual, "source_dual")
            physical(p["source_targets"], d, "source_targets")
        prediction = k @ dual / unit
        alphabet = np.array(read_json(packet / "00/alphabet.json")["values"])
        with np.load(ROOT / spec["query_packet"] / "01/anchors.npz") as p:
            anchors = {
                q: p[f"q{i}_anchor"].copy() for i, q in enumerate(spec["queries"])
            }
        with np.load(packet / "02/predictions.npz") as p:
            saved = {n: p[n].copy() for n in p.files}
        n = len(meta)
        yn, yp, scalar = (np.empty(n) for _ in range(3))
        parts, nparts, motions, nmotions = (
            np.empty((n, size)) for size in (3, 3, 10, 10)
        )
        old = read_json(context / "01/functions.json")
        for qi, q in enumerate(spec["queries"]):
            with np.load(context / old[q]["receiver_tensor"]["path"]) as p:
                old_pred = k @ p["weighted_dual"] / unit
            for r in (v for v in meta if v["query"] == q):
                i, s = r["row"], r["state_index"]
                zr, zt = (
                    np.array(read_json(ROOT / r[name])["native"]["candidate_logits"])
                    for name in ("reference_path", "unit_path")
                )
                pr, pt = probability(zr), probability(zt)
                delta = prediction[s, qi * 10 : (qi + 1) * 10]
                dn = (zt - zt.mean()) - (zr - zr.mean())
                yn[i], yp[i] = (
                    alphabet @ (pt - pr),
                    alphabet @ (probability(zr + delta) - pr),
                )
                g = pr * (alphabet - alphabet @ pr)
                b, l, bn, ln = anchors[q] @ delta, g @ delta, anchors[q] @ dn, g @ dn
                parts[i], nparts[i] = [b, l - b, yp[i] - l], [bn, ln - bn, yn[i] - ln]
                motions[i], nmotions[i], scalar[i] = delta, dn, old_pred[s]
        for name, expected in (
            ("source_order_prediction", prediction),
            ("predicted_delta", motions),
            ("native_delta", nmotions),
            ("predicted_parts", parts),
            ("native_parts", nparts),
            ("native", yn),
            ("predicted", yp),
            ("scalar_receiver", scalar),
        ):
            physical(saved[name], expected, "array_" + name)
        counts["actions"] = n
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
                for r in rr
            }
            for name, rr in tables.items()
        }
        lookup = {(r["state_index"], r["query"]): r["row"] for r in meta}
        idx = 0
        with np.load(packet / "02/remainder_covariance_blocks.npz") as blocks:
            for role, design in itertools.product(
                spec["roles"],
                ("all_states", *sorted({r["arm"] for r in first_rows.values()})),
            ):
                selected = [
                    r
                    for _, r in sorted(first_rows.items())
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
                    ids = np.array([lookup[s, q] for s in ss])
                    fv = []
                    for f in sorted(set(fs)):
                        mask = np.array([v == f for v in fs])
                        value = float(
                            np.mean(
                                (yp[ids][mask] - yn[ids][mask]) ** 2
                                - (scalar[ids][mask] - yn[ids][mask]) ** 2
                            )
                        )
                        fv.append(value)
                        table(
                            indexed["family_comparisons"][role, design, q, str(f)],
                            {
                                **info,
                                "query": q,
                                "family": f,
                                "observations": int(mask.sum()),
                                "mse_delta_vector_minus_scalar": value,
                            },
                            "families",
                        )
                    fv = np.array(fv)
                    sample = np.random.default_rng(spec["bootstrap_seed"]).integers(
                        0, len(fv), (spec["bootstrap_replicates"], len(fv))
                    )
                    ci = np.quantile(fv[sample].mean(axis=1), [0.025, 0.975])
                    vm, sm = (
                        metrics(yn[ids], yp[ids], ww),
                        metrics(yn[ids], scalar[ids], ww),
                    )
                    mse = float(
                        ww @ np.sum((motions[ids] - nmotions[ids]) ** 2, axis=1)
                    )
                    norm = float(ww @ np.sum(nmotions[ids] ** 2, axis=1))
                    table(
                        indexed["action_statistics"][role, design, q, ""],
                        {
                            **info,
                            "query": q,
                            **{f"vector_{k}": v for k, v in vm.items()},
                            **{f"scalar_{k}": v for k, v in sm.items()},
                            "motion_rmse": np.sqrt(mse),
                            "motion_relative_rmse": np.sqrt(mse / norm)
                            if norm
                            else None,
                            "mse_delta_mean": fv.mean(),
                            "mse_delta_ci_low": ci[0],
                            "mse_delta_ci_high": ci[1],
                            "positive_gain_families": int(sum(fv < 0)),
                        },
                        "statistics",
                    )
                    physical(
                        float(
                            indexed["action_statistics"][role, design, q, ""][
                                "motion_relative_rmse"
                            ]
                        ),
                        np.sqrt(mse / norm),
                        "motion_ratio",
                    )
                for qa, qb in itertools.combinations(spec["queries"], 2):
                    ia, ib = (np.array([lookup[s, q] for s in ss]) for q in (qa, qb))
                    key = {**info, "query_a": qa, "query_b": qb}
                    pa, pb, ra, rb = yp[ia], yp[ib], yn[ia] - yp[ia], yn[ib] - yp[ib]
                    terms = {
                        nn: covariance(a, b, ww)
                        for nn, a, b in (
                            ("PP", pa, pb),
                            ("PR", pa, rb),
                            ("RP", ra, pb),
                            ("RR", ra, rb),
                        )
                    }
                    nc, oc = (
                        covariance(yn[ia], yn[ib], ww),
                        covariance(scalar[ia], scalar[ib], ww),
                    )
                    table(
                        indexed["joint_covariance"][role, design, qa, qb],
                        {
                            **key,
                            **terms,
                            "native_covariance": nc,
                            "scalar_covariance": oc,
                            "recombination_error": abs(nc - sum(terms.values())),
                            "vector_covariance_gap": nc - terms["PP"],
                            "scalar_covariance_gap": nc - oc,
                            "vector_native_sign_agree": bool(
                                np.sign(nc) == np.sign(terms["PP"])
                            ),
                            "scalar_native_sign_agree": bool(
                                np.sign(nc) == np.sign(oc)
                            ),
                        },
                        "joint",
                    )
                    covs = []
                    for a, b in (
                        (motions[ia], motions[ib]),
                        (motions[ia], nmotions[ib] - motions[ib]),
                        (nmotions[ia] - motions[ia], motions[ib]),
                        (nmotions[ia] - motions[ia], nmotions[ib] - motions[ib]),
                    ):
                        covs.append(
                            np.einsum("i,ij,ik->jk", ww, a - ww @ a, b - ww @ b)
                        )
                    physical(blocks["blocks"][idx], np.array(covs), "matrix_blocks")
                    table(
                        indexed["matrix_addresses"][role, design, qa, qb],
                        {**key, "matrix_row": idx},
                        "matrix_addresses",
                    )
                    expected = {**key}
                    for prefix, array, original in (
                        ("predicted", parts, saved["predicted_parts"]),
                        ("native", nparts, saved["native_parts"]),
                    ):
                        ac = accounts(array[ia], array[ib], ww)
                        expected.update({f"{prefix}_{k}": v for k, v in ac.items()})
                        ea = np.max(abs(array[ia] - original[ia]), axis=0)
                        eb = np.max(abs(array[ib] - original[ib]), axis=0)
                        term_bounds = {
                            xx + yy: covariance_bound(
                                array[ia, j], array[ib, l], ea[j], eb[l], ww
                            )
                            for j, xx in enumerate("BGC")
                            for l, yy in enumerate("BGC")
                        }
                        ed = covariance_bound(
                            array[ia].sum(axis=1),
                            array[ib].sum(axis=1),
                            ea.sum(),
                            eb.sum(),
                            ww,
                        )
                        for label, names in (
                            ("sensitivity", ("BG", "GB", "GG")),
                            ("link", ("BC", "GC", "CB", "CG", "CC")),
                        ):
                            num, den = ac[label + "_increment"], ac["native_covariance"]
                            en = sum(term_bounds[name] for name in names)
                            en += (
                                64
                                * len(ww)
                                * np.finfo(float).eps
                                * sum(
                                    abs(ac[name]) for name in ("BB", "BG", "GB", "GG")
                                )
                            )
                            bound = ratio_bound(num, den, en, ed)
                            field = f"{prefix}_{label}_native_ratio"
                            actual = indexed["account_covariance"][
                                role, design, qa, qb
                            ][field]
                            if not actual:
                                if den != 0:
                                    raise ValueError("missing nonzero ratio")
                                continue
                            difference = abs(float(actual) - num / den)
                            point_pass = bool(
                                np.isclose(
                                    float(actual),
                                    num / den,
                                    atol=spec["arithmetic_absolute_tolerance"],
                                    rtol=spec["aggregate_relative_tolerance"],
                                )
                            )
                            if bound is not None and difference > bound:
                                raise ValueError(
                                    f"outside analytical ratio bound {key}/{field}"
                                )
                            ratios.append(
                                {
                                    **key,
                                    "field": field,
                                    "primary": float(actual),
                                    "alternate": num / den,
                                    "denominator": den,
                                    "numerator_error_bound": en,
                                    "denominator_error_bound": ed,
                                    "ratio_error_bound": bound,
                                    "difference": difference,
                                    "original_point_check_pass": point_pass,
                                }
                            )
                    expected.update(
                        {
                            f"motion_{nn}_fixed_contraction": anchors[qa]
                            @ covs[j]
                            @ anchors[qb]
                            for j, nn in enumerate(("PP", "PR", "RP", "RR"))
                        }
                    )
                    table(
                        indexed["account_covariance"][role, design, qa, qb],
                        expected,
                        "accounts",
                    )
                    idx += 1
        required = Counter(
            actions=2172,
            families=354,
            statistics=72,
            joint=72,
            accounts=72,
            matrix_addresses=72,
        )
        if counts != required or len(ratios) != 288:
            raise ValueError(f"incomplete companion {counts}/{len(ratios)}")
        write_csv(out / "ratio_conditioning.csv", ratios)
        write_json(
            out / "summary.json",
            {
                "status": "PHYSICAL_UNITS_VERIFIED_ORIGINAL_RATIO_POINT_FAILURE_RETAINED",
                "counts": dict(counts),
                "ratio_rows": len(ratios),
                "original_point_failures": sum(
                    not r["original_point_check_pass"] for r in ratios
                ),
                "max_errors": errors,
                "max_ratio_difference": max(r["difference"] for r in ratios),
                "original_failure_sha256": digest(FIRST / "failure.json"),
                "verifier_sha256": digest(Path(__file__)),
                "new_model_calls": 0,
                "new_fits": 0,
                "primary_results_changed": False,
                "ratios_are_conditioned_diagnostics_not_precise_effect_shares": True,
                "same_agent_alternate_arithmetic_not_independent_replication": True,
            },
        )
        print(read_json(out / "summary.json"), flush=True)
    except BaseException as exc:
        write_json(
            out / "failure.json",
            {"error": repr(exc), "counts": dict(counts), "max_errors": errors},
        )
        raise


def main():
    """Run one companion in a fresh namespace, without modifying old checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    collector_guard(ROOT)
    verify(args.packet, args.out)


if __name__ == "__main__":
    main()

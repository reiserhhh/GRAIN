#!/usr/bin/env python3
"""One finite choice/geometry mechanism package, not a prediction grid."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_choice_geometry as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_choice_geometry_hm188.json"
DEFAULT = ROOT / "results/grain_math_response_v1/choice_geometry_response_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CHOICE_GEOMETRY_PLAN_HM188.md",
    "grain_research/construction/math_choice_geometry.py",
    "scripts/run_grain_choice_geometry_hm188.py",
    "scripts/score_grain_choice_geometry_cold.py",
    "tests/test_grain_choice_geometry.py",
)


def prepare(out):
    """Freeze identical numeric events and inherited assignments before new analysis."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    inherited = R.check(
        parent, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")
    )
    z = R.prepared(parent)
    if z["coordinates"].shape != (386, 2, 12, 15):
        raise ValueError("Changed SAME-event source scope")
    out.mkdir(parents=True, exist_ok=False)
    files = (
        "manifest.json",
        "observations.npz",
        "construction_accounts.npz",
        "prepared_seal.json",
        "compiled_seal.json",
        "scored_seal.json",
        "verification_seal.json",
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": inherited["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in files
            },
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **z)
    R.write(
        out / "quality.json",
        {
            "authors": len(z["labels"]),
            "events": int(z["conditions"].size),
            "fold_counts": np.bincount(z["folds"]).tolist(),
            "cold_input": R.read(parent / "quality.json")["cold_input"],
            "source_parent": cfg["parent"],
            "no_new_neural_private_API_training": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "quality.json"))
    print(json.dumps(R.read(out / "quality.json")))


def native_fields(m, y, folds):
    """Construct complete accounts with all original target labels and support."""
    return M.fields(m["a"], m["geometry"], m["support"], y, folds)


def values(fields):
    """Extract complete ordered target statistics with no favorable selection."""
    return np.asarray([r["signed_product"] for r in M.statistics(fields)])


def primary(stats):
    """Registered all-target contrast is total-product, not coupling-only norm."""
    full = next(
        r["signed_product"]
        for r in stats
        if (r["kind"], r["target"]) == ("full", "all")
    )
    product = next(
        r["signed_product"]
        for r in stats
        if (r["kind"], r["target"]) == ("product", "all")
    )
    return {"full": full, "product": product, "difference": full - product}


def compile_source(out):
    """Run fixed algebra and pairing references once, retaining every cross term."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.monotonic()
    parent = ROOT / cfg["parent"]
    with np.load(parent / "construction_accounts.npz", allow_pickle=False) as a:
        yp, cp, inherited_D = (
            a["label_permutations"],
            a["pairing_permutations"],
            a["operators"][:, :, 2],
        )
    m = M.observations(z["coordinates"], z["conditions"])
    result = native_fields(m, z["labels"], z["folds"])
    f, stats = result["fields"], M.statistics(result["fields"])
    n = cfg["null_replicates"]
    nulls = np.empty((3, n, len(stats)))
    orders, diagnostics = [], []
    rng = np.random.default_rng(cfg["coupling_seed"])
    for r in range(n):
        if time.monotonic() - start > cfg["max_compile_seconds"]:
            R.write(
                out / "compile_failure.json",
                {"reason": "fixed_CPU_cap", "completed": r},
            )
            raise RuntimeError("Fixed cap; do not restart or enlarge")
        order, diag = M.coupling_permutation(z["labels"], z["folds"], m["support"], rng)
        orders.append(order)
        diagnostics.append(diag)
        swapped = M.fields(
            m["a"][order], m["geometry"], m["support"], z["labels"], z["folds"]
        )["fields"]
        nulls[0, r] = values(swapped)
        nulls[1, r] = values(native_fields(m, yp[r], z["folds"])["fields"])
        relinked = M.observations(z["coordinates"], cp[r])
        nulls[2, r] = values(native_fields(relinked, z["labels"], z["folds"])["fields"])
    rows = []
    for i, row in enumerate(stats):
        record = row.copy()
        for k, name in enumerate(("coupling", "profile", "condition")):
            ref = nulls[k, :, i]
            invariant = name == "coupling" and row["kind"] in (
                "choice",
                "geometry",
                "product",
            )
            record[name + "_null_mean"] = float(ref.mean())
            record[name + "_range"] = float(np.ptp(ref))
            record[name + "_upper_tail"] = (
                None
                if invariant or np.ptp(ref) == 0
                else float((1 + np.sum(ref >= row["signed_product"])) / (n + 1))
            )
            record[name + "_invariant"] = invariant or np.ptp(ref) == 0
        rows.append(record)
    full_index = next(
        i for i, r in enumerate(stats) if (r["kind"], r["target"]) == ("full", "all")
    )
    product_index = next(
        i for i, r in enumerate(stats) if (r["kind"], r["target"]) == ("product", "all")
    )
    gap_null = nulls[:, :, full_index] - nulls[:, :, product_index]
    decision = primary(stats)
    for k, name in enumerate(("coupling", "profile", "condition")):
        decision[name + "_null_mean"] = float(gap_null[k].mean())
        decision[name + "_range"] = float(np.ptp(gap_null[k]))
        decision[name + "_upper_tail"] = (
            None
            if np.ptp(gap_null[k]) == 0
            else float((1 + np.sum(gap_null[k] >= decision["difference"])) / (n + 1))
        )
    pooled = native_fields(m, z["labels"], None)["fields"][0]
    refs = [
        M.reference(m["a"][:, v], m["geometry"][:, v], m["support"][:, v])
        for v in (0, 1)
    ]
    calibration = {
        k: np.asarray([r[k] for r in refs]) for k in ("q", "A", "G", "L", "f", "total")
    }
    R.archive(
        out / "reader_bundle.npz",
        version=np.asarray(M.VERSION),
        fields=pooled[:, :, 4],
        basis=z["basis"],
        mapping=z["mapping"],
        **calibration,
    )
    R.archive(
        out / "construction_accounts.npz",
        a=m["a"],
        geometry=m["geometry"],
        support=m["support"],
        weighted=m["weighted"],
        fields=f,
        pooled_fields=pooled,
        nulls=nulls,
        orders=np.asarray(orders),
        gap_null=gap_null,
        **{"reference_" + k: v for k, v in calibration.items()},
    )
    terms = M.cross_terms(f)
    receipt = {
        "primary": decision,
        "seconds": time.monotonic() - start,
        "inherited_D_error": float(np.max(np.abs(m["weighted"] - inherited_D))),
        "four_field_reconstruction": float(
            np.max(np.abs(f[:, :, :, :4].sum(3) - f[:, :, :, 4]))
        ),
        "cross_term_reconstruction": abs(
            sum(r["signed_product"] for r in terms if r["target"] == "all")
            - decision["full"]
        ),
        "coupling_groups": diagnostics[0]["groups"],
        "singleton_groups": diagnostics[0]["singleton_groups"],
        "movable_authors": diagnostics[0]["movable_authors"],
        "fixed_points_range": [
            min(r["fixed_points"] for r in diagnostics),
            max(r["fixed_points"] for r in diagnostics),
        ],
        "all_source_authors_retained": len(m["a"]),
        "mixed_support_by_view": m["support"].sum(0).tolist(),
        "invariant_coupling_comparisons": ["choice", "geometry", "product"],
        "same_agent_not_blind": True,
        "new_neural_API_private_training": 0,
        "references_exploratory_not_causal_or_accuracy": True,
    }
    if (
        max(
            receipt[k]
            for k in (
                "inherited_D_error",
                "four_field_reconstruction",
                "cross_term_reconstruction",
            )
        )
        > cfg["comparison_tolerance"]
    ):
        R.write(out / "compile_failure.json", receipt)
        raise ValueError("Account mismatch; no precision rescue")
    R.table(out / "target_comparison.csv", rows)
    R.table(out / "cross_terms.csv", terms)
    R.table(out / "source_coverage.csv", result["coverage"])
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "reader_bundle.npz",
            "construction_accounts.npz",
            "target_comparison.csv",
            "cross_terms.csv",
            "source_coverage.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Start one isolated numeric scoring CLI; never renew a failed experiment."""
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_choice_geometry_cold.py"),
            "--bundle",
            str(out / "reader_bundle.npz"),
            "--input",
            str(ROOT / R.read(out / "quality.json")["cold_input"]),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        R.write(
            out / "cold_failure.json",
            {"returncode": result.returncode, "stderr": result.stderr},
        )
        raise RuntimeError("Cold failure retained; no retry")
    receipt = json.loads(result.stdout)
    if receipt["frames"] != cfg["cold_frames"]:
        raise ValueError("Changed cold scope")
    R.write(out / "cold_receipt.json", receipt)
    R.seal(out, "scored_seal.json", ("cold_output.jsonl", "cold_receipt.json"))
    print(json.dumps(receipt))


def literal_observations(x, conditions):
    """Use conditional means directly, rather than rho divided by opportunity."""
    g = np.full((*x.shape[:2], x.shape[-1], x.shape[-1]), np.nan)
    s = np.zeros(x.shape[:2], bool)
    a = np.empty(x.shape[:2])
    d = np.zeros_like(g)
    for u in range(len(x)):
        for v in (0, 1):
            c, row = conditions[u, v], x[u, v]
            p = float(c.mean())
            a[u, v] = (1 - 2 * p) * p * (1 - p)
            s[u, v] = 0 < p < 1
            if s[u, v]:
                delta = row[c == 1].mean(0) - row[c == 0].mean(0)
                g[u, v] = np.outer(delta, delta)
                d[u, v] = a[u, v] * g[u, v]
    return {"a": a, "geometry": g, "support": s, "weighted": d}


def literal_fields(m, labels, folds):
    """Reconstruct contrasts from raw group means and uncentered products."""
    result = np.empty((2, 2, 5, 7, 15, 15))
    for b in (0, 1):
        for v in (0, 1):
            for target in range(5):
                sides = []
                for side in (0, 1):
                    group = (folds == b) & (labels[:, target] == side)
                    ids = group & m["support"][:, v]
                    q = float(ids.sum() / group.sum())
                    a, g = m["a"][ids, v], m["geometry"][ids, v]
                    A, G = a.mean(), g.mean(0)
                    f = np.mean(a[:, None, None] * g, axis=0)
                    sides.append(
                        {
                            "q": q,
                            "A": A,
                            "G": G,
                            "f": f,
                            "L": f - A * G,
                            "total": m["weighted"][group, v].mean(0),
                            "product": q * A * G,
                        }
                    )
                low, high = sides
                q, a = (low["q"] + high["q"]) / 2, (low["A"] + high["A"]) / 2
                result[b, v, target, 0] = (
                    (high["q"] - low["q"]) * (high["f"] + low["f"]) / 2
                )
                result[b, v, target, 1] = (
                    q * (high["A"] - low["A"]) * (high["G"] + low["G"]) / 2
                )
                result[b, v, target, 2] = q * a * (high["G"] - low["G"])
                result[b, v, target, 3] = q * (high["L"] - low["L"])
                result[b, v, target, 4] = high["total"] - low["total"]
                result[b, v, target, 5] = high["product"] - low["product"]
                result[b, v, target, 6] = (
                    result[b, v, target, 4] - result[b, v, target, 5]
                )
    return result


def verify(out):
    """Replay complete source, pairing references and cold scalar accounts independently."""
    cfg, z = (
        R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"],
        R.prepared(out),
    )
    with np.load(out / "construction_accounts.npz", allow_pickle=False) as a:
        saved = {k: a[k] for k in a.files}
    with np.load(
        ROOT / cfg["parent"] / "construction_accounts.npz", allow_pickle=False
    ) as a:
        yp, cp = a["label_permutations"], a["pairing_permutations"]
    m = literal_observations(z["coordinates"], z["conditions"])
    fields = literal_fields(m, z["labels"], z["folds"])
    errors = {
        "source_weighted": float(np.max(np.abs(m["weighted"] - saved["weighted"]))),
        "source_fields": float(np.max(np.abs(fields - saved["fields"]))),
        "null_statistics": 0.0,
        "null_marginal_invariance": 0.0,
        "cold": 0.0,
        "cold_account": 0.0,
        "cross_terms": 0.0,
        "original_coordinates": 0.0,
    }
    signature = m["support"].astype(int) @ np.array([1, 2])
    for r in range(cfg["null_replicates"]):
        order = saved["orders"][r]
        if not (
            np.array_equal(z["labels"][order], z["labels"])
            and np.array_equal(z["folds"][order], z["folds"])
            and np.array_equal(signature[order], signature)
        ):
            raise ValueError("Changed declared pairing strata")
        swapped = m | {
            "a": m["a"][order],
            "weighted": np.where(
                m["support"][..., None, None],
                m["a"][order, :, None, None] * m["geometry"],
                0,
            ),
        }
        banks = (
            literal_fields(swapped, z["labels"], z["folds"]),
            literal_fields(m, yp[r], z["folds"]),
            literal_fields(
                literal_observations(z["coordinates"], cp[r]), z["labels"], z["folds"]
            ),
        )
        for k, bank in enumerate(banks):
            errors["null_statistics"] = max(
                errors["null_statistics"],
                float(np.max(np.abs(values(bank) - saved["nulls"][k, r]))),
            )
        errors["null_marginal_invariance"] = max(
            errors["null_marginal_invariance"],
            float(
                np.max(
                    np.abs(banks[0][:, :, :, [1, 2, 5]] - fields[:, :, :, [1, 2, 5]])
                )
            ),
        )
    terms = M.cross_terms(fields)
    full = primary(M.statistics(fields))["full"]
    errors["cross_terms"] = abs(
        sum(t["signed_product"] for t in terms if t["target"] == "all") - full
    )
    with np.load(out / "reader_bundle.npz", allow_pickle=False) as a:
        bundle = {k: a[k] for k in a.files}
    frames = missing = scalars = 0
    path = ROOT / R.read(out / "quality.json")["cold_input"]
    with path.open() as source, (out / "cold_output.jsonl").open() as sink:
        for line, output in zip(source, sink, strict=True):
            row, got = json.loads(line), json.loads(output)
            if row["frame_key"] != got["frame_key"]:
                raise ValueError("Cold address mismatch")
            h = np.array([e["vector"] for e in row["events"]])
            c = np.array([e["condition"] for e in row["events"]])
            # Adapter duplication is not two independent observed texts.
            lm = literal_observations(
                np.tile(h @ bundle["basis"], (1, 2, 1, 1)), np.tile(c, (1, 2, 1))
            )
            s, a, g = bool(lm["support"][0, 0]), lm["a"][0, 0], lm["geometry"][0, 0]
            if (
                got["conditional_support"] != s
                or got["counts"] != [int((c == j).sum()) for j in (0, 1)]
                or got["dispersion_support"] != (min(got["counts"]) >= 2)
            ):
                raise ValueError("Cold availability mismatch")
            for v in (0, 1):
                q, A, G, L, f = (bundle[k][v] for k in ("q", "A", "G", "L", "f"))
                parts = [
                    (int(s) - q) * f,
                    (a - A) * G if s else np.zeros_like(f),
                    A * (g - G) if s else np.zeros_like(f),
                    (a - A) * (g - G) - L if s else np.zeros_like(f),
                ]
                for k, name in enumerate(M.COMPONENTS):
                    expected = np.sum(bundle["fields"][v] * parts[k], axis=(-1, -2))
                    errors["cold"] = max(
                        errors["cold"],
                        float(np.max(np.abs(expected - got["readings"][name][v]))),
                    )
                    scalars += expected.size
                D = a * g if s else np.zeros_like(f)
                p = float(c.mean())
                matrices = {
                    "relative_full": D - q * f,
                    "source_reference": q * f,
                    "weighted_displacement": p * (1 - p) * g if s else np.zeros_like(f),
                    "choice_displacement": D,
                }
                if s:
                    matrices["conditional_geometry"] = g
                elif got["conditional_geometry"] is not None:
                    raise ValueError("Absent geometry imputed")
                for name, matrix in matrices.items():
                    expected = np.sum(bundle["fields"][v] * matrix, axis=(-1, -2))
                    errors["cold"] = max(
                        errors["cold"], float(np.max(np.abs(expected - got[name][v])))
                    )
                    scalars += expected.size
            reconstructed = np.array(
                [got["readings"][name] for name in M.COMPONENTS]
            ).sum(0)
            errors["cold_account"] = max(
                errors["cold_account"],
                float(np.max(np.abs(reconstructed - got["relative_full"]))),
            )
            errors["original_coordinates"] = max(
                errors["original_coordinates"],
                float(
                    np.max(
                        np.abs(
                            (h @ bundle["mapping"]).mean(0) - got["original_GRAIN_mean"]
                        )
                    )
                ),
            )
            frames += 1
            missing += not s
    receipt = {
        "frames": frames,
        "scalar_readings": scalars,
        "missing_geometry": missing,
        "errors": errors,
        "same_agent_not_blind": True,
    }
    if (
        frames != cfg["cold_frames"]
        or max(errors.values()) > cfg["comparison_tolerance"]
    ):
        R.write(out / "verification_failure.json", receipt)
        raise ValueError("Fixed replay failed; no precision rescue")
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt))


def main():
    """Execute exactly one stage, with immutable output and no autonomous next queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compile", "score", "verify"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compile": compile_source, "score": score, "verify": verify}[
        args.stage
    ](args.output_dir)


if __name__ == "__main__":
    main()

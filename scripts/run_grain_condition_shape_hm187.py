#!/usr/bin/env python3
"""Single-write conditional shape construction with fixed competing mechanisms."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_condition_pairing as P
from grain_research.construction import math_condition_shape as M
from grain_research.construction import math_target_complement as C
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_condition_shape_hm187.json"
DEFAULT = ROOT / "results/grain_math_response_v1/condition_shape_response_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CONDITION_SHAPE_PLAN_HM187.md",
    "grain_research/construction/math_condition_shape.py",
    "scripts/run_grain_condition_shape_hm187.py",
    "scripts/score_grain_condition_shape_cold.py",
    "tests/test_grain_condition_shape.py",
    "grain_research/construction/math_condition_pairing.py",
    "grain_research/construction/math_target_complement.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def literal_moments(z, c):
    """Replay each covariance directly from conditional groups, not the cross-moment formula."""
    z, c = np.asarray(z, float), np.asarray(c)
    d = z.shape[-1]
    pooled, raw, drift, shape = [], [], [], []
    for row, condition in zip(
        z.reshape(-1, z.shape[-2], d), c.reshape(-1, c.shape[-1]), strict=True
    ):
        p = float(condition.mean())
        mu = row.mean(0)
        covs, means = [], []
        for value in (0, 1):
            values = row[condition == value]
            if len(values):
                center = values - values.mean(0)
                covs.append(sum(np.outer(v, v) for v in center) / len(values))
                means.append(values.mean(0))
            else:
                covs.append(np.zeros((d, d)))
                means.append(np.zeros(d))
        pooled.append(sum(np.outer(v - mu, v - mu) for v in row) / len(row))
        if 0 < p < 1:
            delta = means[1] - means[0]
            shape.append(p * (1 - p) * (covs[1] - covs[0]))
            drift.append(p * (1 - p) * (1 - 2 * p) * np.outer(delta, delta))
        else:
            shape.append(np.zeros((d, d)))
            drift.append(np.zeros((d, d)))
        raw.append(shape[-1] + drift[-1])
    return np.stack((pooled, raw, drift, shape), axis=1).reshape(*z.shape[:-2], 4, d, d)


def prepare(out):
    """Freeze inherited numeric event identity and fixed-span metric before new comparisons."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    inherited = R.check(
        parent, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")
    )
    z = R.prepared(parent)
    q, _ = C.fixed_basis(z["mapping"])
    if z["vectors"].shape != (386, 2, 12, 1024) or q.shape != (1024, 15):
        raise ValueError("Changed complete inherited numerical scope")
    out.mkdir(parents=True, exist_ok=False)
    inputs = {
        str((parent / p).relative_to(ROOT)): R.digest(parent / p)
        for p in (
            "manifest.json",
            "observations.npz",
            "cold_input.jsonl",
            "prepared_seal.json",
            "compiled_seal.json",
            "scored_seal.json",
            "verification_seal.json",
        )
    }
    code = inherited["code"] | {p: R.digest(ROOT / p) for p in SCIENCE}
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": code,
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        coordinates=z["vectors"] @ q,
        conditions=z["conditions"],
        labels=z["labels"],
        folds=z["folds"],
        source_indices=z["source_indices"],
        basis=q,
        mapping=z["mapping"],
    )
    quality = {
        "source_authors": len(z["labels"]),
        "events": int(np.prod(z["conditions"].shape)),
        "author_fold_counts": np.bincount(z["folds"]).tolist(),
        "orthonormal_error": float(np.max(np.abs(q.T @ q - np.eye(15)))),
        "cold_input": str((parent / "cold_input.jsonl").relative_to(ROOT)),
        "space": cfg["space"],
        "source_view_rows_not_independent_authors": True,
    }
    R.write(out / "quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "quality.json"))
    print(json.dumps(quality))


def compile_source(out):
    """Run only the fixed operator and identity/target references, never a precision search."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.monotonic()
    m = M.moments(z["coordinates"], z["conditions"])
    x = m["operators"]
    pooled = M.fields(x, z["labels"])["fields"][0]
    crossed = M.fields(x, z["labels"], z["folds"])["fields"]
    stats = M.statistics(crossed)
    n = cfg["null_replicates"]
    label_null, pairing_null = np.empty((n, len(stats))), np.empty((n, len(stats)))
    label_permutations = np.empty((n, *z["labels"].shape), z["labels"].dtype)
    pairing_permutations = np.empty((n, *z["conditions"].shape), z["conditions"].dtype)
    author_permutations = np.empty((n, len(x)), int)
    author_null = np.empty((n, 4))
    author_refs = [
        M.author_comparison(x[:, :, k], m["counts"][:, 1, 1]) for k in range(4)
    ]
    author_grams = [
        x[:, 0, k].reshape(len(x), -1) @ x[:, 1, k].reshape(len(x), -1).T
        for k in range(4)
    ]
    rngs = [
        np.random.default_rng(cfg[key])
        for key in ("author_null_seed", "profile_null_seed", "pairing_null_seed")
    ]
    invariant_error = 0.0
    for r in range(n):
        if time.monotonic() - start > cfg["max_compile_seconds"]:
            R.write(
                out / "compile_failure.json",
                {"reason": "fixed_CPU_cap", "completed_references": r},
            )
            raise RuntimeError("Fixed construction cap reached; no restart")
        order = M.permute_authors(m["counts"][:, 1, 1], rngs[0])
        yp = P.permute_profiles(z["labels"], z["folds"], rngs[1])
        cp = P.relink_conditions(z["conditions"], rngs[2])
        for k, (g, ref) in enumerate(zip(author_grams, author_refs, strict=True)):
            author_null[r, k] = np.mean(
                (g[np.arange(len(x)), order] - ref["stranger"])[ref["support"]]
            )
        label_null[r] = [
            v["signed_product"]
            for v in M.statistics(M.fields(x, yp, z["folds"])["fields"])
        ]
        cm = M.moments(z["coordinates"], cp)
        pairing_null[r] = [
            v["signed_product"]
            for v in M.statistics(
                M.fields(cm["operators"], z["labels"], z["folds"])["fields"]
            )
        ]
        invariant_error = max(
            invariant_error,
            float(np.max(np.abs(cm["operators"][:, :, 0] - x[:, :, 0]))),
        )
        author_permutations[r], label_permutations[r], pairing_permutations[r] = (
            order,
            yp,
            cp,
        )
    rows = []
    for i, row in enumerate(stats):
        row = row.copy()
        for name, null in (
            ("profile", label_null[:, i]),
            ("pairing", pairing_null[:, i]),
        ):
            row[name + "_null_mean"] = float(null.mean())
            row[name + "_upper_tail"] = float(
                (1 + np.count_nonzero(null >= row["signed_product"])) / (n + 1)
            )
        rows.append(row)
    author_rows = []
    for k, ref in enumerate(author_refs):
        author_rows.append(
            {
                "kind": M.KINDS[k],
                "primary": ref["primary"],
                "self_mean": float(ref["self"][ref["support"]].mean()),
                "stranger_mean": float(ref["stranger"][ref["support"]].mean()),
                "supported_authors": int(ref["support"].sum()),
                "unsupported_authors": int((~ref["support"]).sum()),
                "null_mean": float(author_null[:, k].mean()),
                "upper_tail": float(
                    (1 + np.count_nonzero(author_null[:, k] >= ref["primary"]))
                    / (n + 1)
                ),
            }
        )
    primary = next(
        row for row in rows if (row["kind"], row["target"]) == ("shape", "all")
    )
    receipt = {
        "author_primary": author_rows[3],
        "target_primary": primary,
        "target_bank_promoted": primary["signed_product"] > 0
        and primary["profile_upper_tail"] <= 0.05
        and primary["pairing_upper_tail"] <= 0.05,
        "conditional_support_by_view": m["support"].sum(0).tolist(),
        "dispersion_support_by_view": m["dispersion_support"].sum(0).tolist(),
        "pooled_relink_invariance_error": invariant_error,
        "null_replicates_each": n,
        "seconds": time.monotonic() - start,
        "new_neural_API_training": 0,
        "secondary_comparisons": "all_retained_unadjusted_descriptive",
    }
    R.archive(
        out / "reader_bundle.npz",
        version=np.asarray(M.VERSION),
        fields=pooled,
        basis=z["basis"],
        mapping=z["mapping"],
    )
    R.archive(
        out / "construction_accounts.npz",
        operators=x,
        fold_fields=crossed,
        counts=m["counts"],
        conditional_delta=m["conditional_shape_delta"],
        fields=pooled,
        label_null=label_null,
        pairing_null=pairing_null,
        author_null=author_null,
        label_permutations=label_permutations,
        pairing_permutations=pairing_permutations,
        author_permutations=author_permutations,
    )
    R.table(out / "target_comparison.csv", rows)
    R.table(out / "author_comparison.csv", author_rows)
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "reader_bundle.npz",
            "construction_accounts.npz",
            "target_comparison.csv",
            "author_comparison.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Run an isolated numeric cold process on the unchanged inherited event stream."""
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    path = ROOT / R.read(out / "quality.json")["cold_input"]
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_condition_shape_cold.py"),
            "--bundle",
            str(out / "reader_bundle.npz"),
            "--input",
            str(path),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        R.write(
            out / "cold_failure.json",
            {"returncode": proc.returncode, "stderr": proc.stderr},
        )
        raise RuntimeError("Cold process failed; no automatic retry")
    receipt = json.loads(proc.stdout)
    if receipt["frames"] != cfg["cold_frames"]:
        raise ValueError("Incomplete inherited cold scope")
    R.write(out / "cold_receipt.json", receipt)
    R.seal(out, "scored_seal.json", ("cold_output.jsonl", "cold_receipt.json"))
    print(json.dumps(receipt))


def verify(out):
    """Use conditional-group formulas for all source,reference and cold accounts."""
    cfg, z = (
        R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"],
        R.prepared(out),
    )
    with np.load(out / "construction_accounts.npz", allow_pickle=False) as a:
        saved = {k: a[k] for k in a.files}
    literal = literal_moments(z["coordinates"], z["conditions"])
    errors = {
        "conditional_group_operators": float(
            np.max(np.abs(literal - saved["operators"]))
        ),
        "fields": 0.0,
        "target_null": 0.0,
        "author_null": 0.0,
        "cold": 0.0,
        "account_reconstruction": 0.0,
        "original_GRAIN_mean": 0.0,
    }
    ref = np.empty_like(saved["fold_fields"])
    for b in (0, 1):
        for q in range(5):
            high = (z["folds"] == b) & (z["labels"][:, q] == 1)
            low = (z["folds"] == b) & (z["labels"][:, q] == 0)
            ref[b, :, :, q] = (literal[high].mean(0) - literal[low].mean(0)).swapaxes(
                0, 1
            )
    errors["fields"] = float(np.max(np.abs(ref - saved["fold_fields"])))
    count = saved["counts"][:, 1, 1]
    stranger = []
    support = []
    for u in range(len(count)):
        ids = np.flatnonzero((count == count[u]) & (np.arange(len(count)) != u))
        support.append(bool(len(ids)))
        stranger.append(
            [
                float(
                    np.mean([np.sum(literal[u, 0, k] * literal[v, 1, k]) for v in ids])
                )
                if len(ids)
                else np.nan
                for k in range(4)
            ]
        )
    stranger, support = np.asarray(stranger), np.asarray(support)
    for r in range(cfg["null_replicates"]):
        yp, cp, order = (
            saved["label_permutations"][r],
            saved["pairing_permutations"][r],
            saved["author_permutations"][r],
        )
        if not np.array_equal(
            cp.sum(-1), z["conditions"].sum(-1)
        ) or not np.array_equal(count[order], count):
            raise ValueError("Reference changed opportunity counts")
        for b in (0, 1):
            mask = z["folds"] == b
            if sorted(map(tuple, yp[mask])) != sorted(map(tuple, z["labels"][mask])):
                raise ValueError("Reference changed joint label dependence")
        alternate = literal_moments(z["coordinates"], cp)
        for ops, labels, expected in (
            (literal, yp, saved["label_null"][r]),
            (alternate, z["labels"], saved["pairing_null"][r]),
        ):
            actual = np.array(
                [
                    v["signed_product"]
                    for v in M.statistics(M.fields(ops, labels, z["folds"])["fields"])
                ]
            )
            errors["target_null"] = max(
                errors["target_null"], float(np.max(np.abs(actual - expected)))
            )
        self_values = np.sum(literal[:, 0] * literal[order, 1], axis=(-1, -2))
        actual_author = (self_values - stranger)[support].mean(0)
        errors["author_null"] = max(
            errors["author_null"],
            float(np.max(np.abs(actual_author - saved["author_null"][r]))),
        )
    pooled = np.stack(
        [
            literal[z["labels"][:, q] == 1].mean(0)
            - literal[z["labels"][:, q] == 0].mean(0)
            for q in range(5)
        ],
        axis=2,
    ).swapaxes(0, 1)
    errors["fields"] = max(
        errors["fields"], float(np.max(np.abs(pooled - saved["fields"])))
    )
    input_path = ROOT / R.read(out / "quality.json")["cold_input"]
    frames = missing = dispersion = scalar_values = 0
    with input_path.open() as source, (out / "cold_output.jsonl").open() as output:
        for line, result in zip(source, output, strict=True):
            row, got = json.loads(line), json.loads(result)
            if row["frame_key"] != got["frame_key"]:
                raise ValueError("Cold identity mismatch")
            h = np.array([e["vector"] for e in row["events"]])
            c = np.array([e["condition"] for e in row["events"]])
            op = literal_moments(h @ z["basis"], c)
            counts = [int((c == v).sum()) for v in (0, 1)]
            supported = min(counts) > 0
            if (
                got["counts"] != counts
                or got["conditional_support"] != supported
                or got["dispersion_support"] != (min(counts) >= 2)
            ):
                raise ValueError("Cold opportunity status mismatch")
            for k, name in enumerate(M.KINDS):
                expected = np.sum(pooled * op[k], axis=(-1, -2))
                errors["cold"] = max(
                    errors["cold"],
                    float(np.max(np.abs(expected - got["readings"][name]))),
                )
                scalar_values += expected.size
            if supported:
                p = float(c.mean())
                expected = np.sum(pooled * (op[3] / (p * (1 - p))), axis=(-1, -2))
                errors["cold"] = max(
                    errors["cold"],
                    float(np.max(np.abs(expected - got["conditional_shape_delta"]))),
                )
                scalar_values += expected.size
                original = h @ z["mapping"]
                expected_delta = original[c == 1].mean(0) - original[c == 0].mean(0)
                errors["original_GRAIN_mean"] = max(
                    errors["original_GRAIN_mean"],
                    float(
                        np.max(
                            np.abs(expected_delta - got["original_GRAIN_mean_delta"])
                        )
                    ),
                )
            else:
                if (
                    got["conditional_shape_delta"] is not None
                    or got["original_GRAIN_mean_delta"] is not None
                ):
                    raise ValueError("Absent conditional contrast imputed")
                missing += 1
            errors["original_GRAIN_mean"] = max(
                errors["original_GRAIN_mean"],
                float(
                    np.max(
                        np.abs((h @ z["mapping"]).mean(0) - got["original_GRAIN_mean"])
                    )
                ),
            )
            reconstruction = (
                np.asarray(got["readings"]["raw_pairing"])
                - np.asarray(got["readings"]["drift"])
                - np.asarray(got["readings"]["shape"])
            )
            errors["account_reconstruction"] = max(
                errors["account_reconstruction"], float(np.max(np.abs(reconstruction)))
            )
            dispersion += got["dispersion_support"]
            frames += 1
    if (
        frames != cfg["cold_frames"]
        or max(errors.values()) > cfg["comparison_tolerance"]
    ):
        raise ValueError("Fixed numerical replay failed; no precision rescue")
    receipt = {
        "frames": frames,
        "operator_scalar_values": scalar_values,
        "missing_conditional_delta": missing,
        "dispersion_supported_frames": dispersion,
        "errors": errors,
        "same_agent_not_blind": True,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt))


def main():
    """Execute one registered stage; never start another packet or overwrite old outputs."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compile", "score", "verify"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = p.parse_args()
    {"prepare": prepare, "compile": compile_source, "score": score, "verify": verify}[
        args.stage
    ](args.output_dir)


if __name__ == "__main__":
    main()

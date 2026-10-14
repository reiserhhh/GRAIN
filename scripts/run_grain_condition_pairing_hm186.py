#!/usr/bin/env python3
"""Construct a finite natural condition-pairing observation, not a precision grid."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_condition_pairing as M
from grain_research.construction import math_target_complement as C
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_condition_pairing_hm186.json"
DEFAULT = ROOT / "results/grain_math_response_v1/condition_pairing_response_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CONDITION_PAIRING_PLAN_HM186.md",
    "grain_research/construction/math_condition_pairing.py",
    "scripts/run_grain_condition_pairing_hm186.py",
    "scripts/score_grain_condition_pairing_cold.py",
    "tests/test_grain_condition_pairing.py",
    "grain_research/construction/math_target_complement.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def prepare(out):
    """Bind old source events and recorded conditions before inspecting new fields."""
    cfg = R.read(ROOT / CONFIG)
    parents = {
        name: ROOT / cfg[name]
        for name in (
            "source_parent",
            "fold_parent",
            "condition_parent",
            "application_parent",
        )
    }
    manifests = {}
    for name, parent in parents.items():
        stages = ()
        if name == "source_parent":
            stages = (
                "compiled_seal.json",
                "scored_seal.json",
                "analysis_seal.json",
                "verification_seal.json",
            )
        elif name == "fold_parent":
            stages = (
                "compiled_seal.json",
                "scored_seal.json",
                "verification_seal.json",
            )
        manifests[name] = R.check(parent, stages)
    source, atlas, conditions, application = [
        R.prepared(parents[name]) for name in parents
    ]
    ids = atlas["source_indices"]
    inverse = np.argsort(conditions["original_units"], axis=1)
    ordered_c = np.take_along_axis(conditions["conditions"], inverse, axis=1)
    ordered_z = np.take_along_axis(
        conditions["coordinates"], inverse[..., None], axis=1
    )
    c = ordered_c[ids, :24].reshape(386, 2, 12)
    x = source["source_vectors"]
    coordinates = ordered_z[ids, :24].reshape(386, 2, 12, 15)
    error = float(np.max(np.abs(coordinates - x @ atlas["mapping"])))
    if (
        x.shape != (386, 2, 12, 1024)
        or not np.array_equal(ids, source["source_indices"])
        or not np.array_equal(ids, application["source_indices"])
        or not np.array_equal(atlas["labels"], source["labels"])
        or not np.array_equal(
            source["panel_indices"], application["application_indices"]
        )
        or np.max(
            np.abs(
                source["panel_vectors"] @ atlas["mapping"] - application["coordinates"]
            )
        )
        > cfg["comparison_tolerance"]
        or set(ids) & set(source["panel_indices"])
        or error > cfg["comparison_tolerance"]
    ):
        raise ValueError("Changed source units or event-condition-coordinate mismatch")
    m = M.moments(x, c)
    code, inputs = {}, {}
    for name, manifest in manifests.items():
        code.update(manifest["code"])
        for filename in ("manifest.json", "observations.npz", "prepared_seal.json"):
            p = parents[name] / filename
            inputs[str(p.relative_to(ROOT))] = R.digest(p)
    inputs[str((parents["source_parent"] / "cold_input.jsonl").relative_to(ROOT))] = (
        R.digest(parents["source_parent"] / "cold_input.jsonl")
    )
    code.update({p: R.digest(ROOT / p) for p in SCIENCE})
    quality = {
        "source_authors": 386,
        "source_views": 2,
        "events_per_view": 12,
        "source_events": int(np.prod(c.shape)),
        "author_fold_counts": np.bincount(atlas["folds"]).tolist(),
        "mixed_condition_support_by_view": m["support"].sum(0).tolist(),
        "mixed_condition_support_both_views": int(m["support"].all(1).sum()),
        "opportunity_range": [
            float(m["opportunity"].min()),
            float(m["opportunity"].max()),
        ],
        "event_coordinate_error": error,
        "source_application_overlap": 0,
        "condition": cfg["condition"],
        "native_same_event_multi_query": False,
        "no_opportunity_policy": "retain_author;conditional_delta_missing_not_zero",
    }
    out.mkdir(parents=True, exist_ok=False)
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
        vectors=x,
        conditions=c,
        labels=atlas["labels"],
        folds=atlas["folds"],
        source_indices=ids,
        mapping=atlas["mapping"],
    )
    frames = 0
    with (
        (parents["source_parent"] / "cold_input.jsonl").open() as old,
        (out / "cold_input.jsonl").open("x") as sink,
    ):
        for line in old:
            row = json.loads(line)
            arm, u, view, subset = map(int, row["frame_key"].split("/"))
            if (
                arm not in (0, 1)
                or u not in range(24)
                or view not in (0, 1)
                or subset not in range(5)
            ):
                raise ValueError("Unknown historical frame address")
            for event in row["events"]:
                if event["event_id"] != f"{u}:{view}:{event['slot']}":
                    raise ValueError("Condition would be attached to another event")
                event["condition"] = int(
                    application["conditions"][u, view, event["slot"]]
                )
            sink.write(json.dumps(row, allow_nan=False) + "\n")
            frames += 1
    if frames != cfg["cold_frames"]:
        raise ValueError("Changed historical cold scope")
    R.write(out / "quality.json", quality | {"cold_frames": frames})
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "quality.json"),
    )
    print(json.dumps(quality))


def compile_source(out):
    """Construct both observation fields and two separately frozen null families."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.monotonic()
    m = M.moments(z["vectors"], z["conditions"])
    basis, _ = C.fixed_basis(z["mapping"])
    pooled = M.source_fields(m["mean"], m["covariance"], z["labels"])
    crossed = M.source_fields(m["mean"], m["covariance"], z["labels"], z["folds"])
    fields = C.split_fields(pooled["fields"][0], basis)
    fold_fields = C.split_fields(crossed["fields"], basis)
    stats = M.statistics(fold_fields)
    n = cfg["null_replicates"]
    label_null, pairing_null = np.empty((n, len(stats))), np.empty((n, len(stats)))
    label_permutations = np.empty((n, *z["labels"].shape), z["labels"].dtype)
    pairing_permutations = np.empty((n, *z["conditions"].shape), z["conditions"].dtype)
    labels_rng = np.random.default_rng(cfg["profile_null_seed"])
    pairing_rng = np.random.default_rng(cfg["pairing_null_seed"])
    invariant_error = 0.0
    for replicate in range(n):
        y = M.permute_profiles(z["labels"], z["folds"], labels_rng)
        f = M.source_fields(m["mean"], m["covariance"], y, z["folds"])["fields"]
        label_null[replicate] = [
            r["signed_inner_product"] for r in M.statistics(C.split_fields(f, basis))
        ]
        c = M.relink_conditions(z["conditions"], pairing_rng)
        cm = M.moments(z["vectors"], c)
        invariant_error = max(
            invariant_error,
            float(np.max(np.abs(cm["p"] - m["p"]))),
            float(np.max(np.abs(cm["mean"] - m["mean"]))),
        )
        f = M.source_fields(cm["mean"], cm["covariance"], z["labels"], z["folds"])[
            "fields"
        ]
        pairing_null[replicate] = [
            r["signed_inner_product"] for r in M.statistics(C.split_fields(f, basis))
        ]
        label_permutations[replicate], pairing_permutations[replicate] = y, c
    rows = []
    for i, row in enumerate(stats):
        references = {}
        for name, values in (
            ("label_profile", label_null[:, i]),
            ("condition_pairing", pairing_null[:, i]),
        ):
            references.update(
                {
                    f"{name}_null_mean": float(values.mean()),
                    f"{name}_upper_tail": float(
                        (1 + np.count_nonzero(values >= row["signed_inner_product"]))
                        / (n + 1)
                    ),
                }
            )
        rows.append(row | references)
    primary = next(
        r
        for r in rows
        if (r["kind"], r["target"], r["space"]) == ("pairing", "all", "full")
    )
    choice = np.einsum("bqu,uv->bvq", crossed["weights"], m["p"])
    choice_rows = [
        {
            "target": t,
            "signed_cross_author_product": float(
                choice[0, :, q].mean() * choice[1, :, q].mean()
            ),
            "within_fold_cross_view": float(
                (choice[0, 0, q] * choice[0, 1, q] + choice[1, 0, q] * choice[1, 1, q])
                / 2
            ),
        }
        for q, t in enumerate(M.TARGETS)
    ]
    promoted = primary["signed_inner_product"] > 0 and all(
        primary[k] <= 0.05
        for k in ("label_profile_upper_tail", "condition_pairing_upper_tail")
    )
    R.archive(out / "reader_bundle.npz", version=np.asarray(M.VERSION), fields=fields)
    R.archive(
        out / "construction_accounts.npz",
        fields=fields,
        fold_fields=fold_fields,
        source_mean=m["mean"],
        source_mixed=m["mixed"],
        source_covariance=m["covariance"],
        p=m["p"],
        opportunity=m["opportunity"],
        support=m["support"],
        delta=m["delta"],
        weights=pooled["weights"],
        fold_weights=crossed["weights"],
        counts=crossed["counts"],
        choice_fields=choice,
        label_null=label_null,
        pairing_null=pairing_null,
        label_permutations=label_permutations,
        pairing_permutations=pairing_permutations,
    )
    R.table(out / "cross_author_comparison.csv", rows)
    R.table(out / "choice_comparison.csv", choice_rows)
    receipt = {
        "primary": primary,
        "registered_bank_promoted": promoted,
        "null_replicates_each": n,
        "relink_invariant_error": invariant_error,
        "seconds": time.monotonic() - start,
        "new_neural_API_native_answer_fit": 0,
        "secondary_comparisons": "descriptive_unadjusted_all_retained",
    }
    if invariant_error != 0:
        raise ValueError("Relink altered invariant accounts; preserve failed output")
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "reader_bundle.npz",
            "construction_accounts.npz",
            "cross_author_comparison.csv",
            "choice_comparison.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Run an isolated numeric scorer without access to source tables or native answers."""
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_condition_pairing_cold.py"),
            "--bundle",
            str(out / "reader_bundle.npz"),
            "--input",
            str(out / "cold_input.jsonl"),
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
        raise RuntimeError("Cold scorer failed; no automatic retry")
    receipt = json.loads(proc.stdout)
    if receipt["frames"] != cfg["cold_frames"]:
        raise ValueError("Incomplete cold scope")
    R.write(out / "cold_receipt.json", receipt)
    R.seal(out, "scored_seal.json", ("cold_output.jsonl", "cold_receipt.json"))
    print(json.dumps(receipt))


def verify(out):
    """Replay actual moments,source covectors,null accounts and every cold reading."""
    cfg, z = (
        R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"],
        R.prepared(out),
    )
    with np.load(out / "construction_accounts.npz", allow_pickle=False) as a:
        saved = {k: a[k] for k in a.files}
    errors = {
        name: 0.0
        for name in (
            "source_mean",
            "source_mixed",
            "source_covariance",
            "supported_delta",
            "source_fields",
            "primary",
            "null_statistics",
            "cold_readings",
            "cold_reconstruction",
            "original_factor_reading",
        )
    }
    literal_mu = np.array(
        [[sum(z["vectors"][u, v]) / 12 for v in (0, 1)] for u in range(386)]
    )
    literal_m = np.array(
        [
            [
                sum(z["vectors"][u, v, t] * z["conditions"][u, v, t] for t in range(12))
                / 12
                for v in (0, 1)
            ]
            for u in range(386)
        ]
    )
    literal_rho = literal_m - saved["p"][..., None] * literal_mu
    for key, actual in (
        ("source_mean", literal_mu),
        ("source_mixed", literal_m),
        ("source_covariance", literal_rho),
    ):
        errors[key] = float(np.max(np.abs(actual - saved[key])))
    for u in range(386):
        for v in (0, 1):
            c = z["conditions"][u, v]
            if c.min() != c.max():
                delta = z["vectors"][u, v, c == 1].mean(0) - z["vectors"][
                    u, v, c == 0
                ].mean(0)
                errors["supported_delta"] = max(
                    errors["supported_delta"],
                    float(np.max(np.abs(delta - saved["delta"][u, v]))),
                )
            elif not np.isnan(saved["delta"][u, v]).all():
                raise ValueError("Missing opportunity was imputed")
    basis, triangular = C.fixed_basis(z["mapping"])
    literal_fields = np.empty_like(saved["fold_fields"])
    for b in (0, 1):
        for k, values in enumerate((literal_mu, literal_rho)):
            for v in (0, 1):
                for q in range(5):
                    high = (z["folds"] == b) & (z["labels"][:, q] == 1)
                    low = (z["folds"] == b) & (z["labels"][:, q] == 0)
                    d = values[high, v].mean(0) - values[low, v].mean(0)
                    literal_fields[b, k, v, q] = C.split_fields(d, basis)
    errors["source_fields"] = float(
        np.max(np.abs(literal_fields - saved["fold_fields"]))
    )
    direct_j = (
        sum(
            float(literal_fields[0, 1, v, q, 0] @ literal_fields[1, 1, w, q, 0])
            for v in (0, 1)
            for w in (0, 1)
            for q in range(5)
        )
        / 4
    )
    errors["primary"] = abs(
        direct_j
        - R.read(out / "construction_receipt.json")["primary"]["signed_inner_product"]
    )
    for i in range(cfg["null_replicates"]):
        yp, cp = saved["label_permutations"][i], saved["pairing_permutations"][i]
        for fold in (0, 1):
            mask = z["folds"] == fold
            if sorted(map(tuple, yp[mask])) != sorted(map(tuple, z["labels"][mask])):
                raise ValueError("Profile null altered trait dependence")
        if not np.array_equal(cp.sum(-1), z["conditions"].sum(-1)):
            raise ValueError("Pairing null altered individual condition opportunity")
        cov = (
            np.einsum("uvt,uvtd->uvd", cp, z["vectors"]) / 12
            - saved["p"][..., None] * literal_mu
        )
        for target_y, target_cov, expected in (
            (yp, literal_rho, saved["label_null"][i]),
            (z["labels"], cov, saved["pairing_null"][i]),
        ):
            fs = M.source_fields(literal_mu, target_cov, target_y, z["folds"])["fields"]
            fs = C.split_fields(fs, basis)
            actual = np.array([r["signed_inner_product"] for r in M.statistics(fs)])
            errors["null_statistics"] = max(
                errors["null_statistics"], float(np.max(np.abs(actual - expected)))
            )
    fields = saved["fields"]
    beta = C.coordinate_fields(fields[..., 0, :], basis, triangular)
    frames = missing = values = 0
    with (
        (out / "cold_input.jsonl").open() as source,
        (out / "cold_output.jsonl").open() as output,
    ):
        for line, result in zip(source, output, strict=True):
            row, got = json.loads(line), json.loads(result)
            if row["frame_key"] != got["frame_key"]:
                raise ValueError("Cold event identity changed")
            x = np.array([e["vector"] for e in row["events"]])
            c = np.array([e["condition"] for e in row["events"]])
            p = float(c.mean())
            mu, mixed = (
                x.mean(0),
                sum(e * value for e, value in zip(x, c, strict=True)) / len(x),
            )
            rho = mixed - p * mu
            observables = {"mean": mu, "mixed": mixed, "covariance": rho}
            if 0 < p < 1:
                observables["conditional_delta"] = x[c == 1].mean(0) - x[c == 0].mean(0)
            else:
                if (
                    got["readings"]["conditional_delta"] is not None
                    or got["delta_support"]
                ):
                    raise ValueError("No-opportunity cold score became neutral")
                missing += 1
            for name, xbar in observables.items():
                literal = np.array(
                    [
                        sum(float(d @ e) for e in [xbar])
                        for d in fields.reshape(-1, 1024)
                    ]
                ).reshape(fields.shape[:-1])
                errors["cold_readings"] = max(
                    errors["cold_readings"],
                    float(np.max(np.abs(literal - got["readings"][name]))),
                )
                coord = np.einsum("kvqd,d->kvq", beta, xbar @ z["mapping"])
                errors["original_factor_reading"] = max(
                    errors["original_factor_reading"],
                    float(
                        np.max(
                            np.abs(coord - np.asarray(got["readings"][name])[..., 1])
                        )
                    ),
                )
                values += literal.size
            readings = got["readings"]
            identity = (
                np.asarray(readings["mixed"])
                - p * np.asarray(readings["mean"])
                - np.asarray(readings["covariance"])
            )
            errors["cold_reconstruction"] = max(
                errors["cold_reconstruction"], float(np.max(np.abs(identity)))
            )
            if (
                got["condition_counts"] != [int((c == v).sum()) for v in (0, 1)]
                or got["p"] != p
            ):
                raise ValueError("Cold opportunity metadata changed")
            frames += 1
    if (
        frames != cfg["cold_frames"]
        or max(errors.values()) > cfg["comparison_tolerance"]
    ):
        raise ValueError("Numerical replay failed; no tolerance/field rescue")
    R.write(
        out / "verification.json",
        {
            "frames": frames,
            "scalar_readings": values,
            "no_opportunity_frames": missing,
            "errors": errors,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(R.read(out / "verification.json")))


def main():
    """Execute one single-write stage without restarting old scientific queues."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compile", "score", "verify"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = p.parse_args()
    {"prepare": prepare, "compile": compile_source, "score": score, "verify": verify}[
        args.stage
    ](args.output_dir)


if __name__ == "__main__":
    main()

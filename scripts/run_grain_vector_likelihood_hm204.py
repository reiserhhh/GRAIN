#!/usr/bin/env python3
"""Compile one positive source-vector measurement law, with no answer fitting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_same_event_multitarget as C
from grain_research.construction import math_vector_likelihood as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_vector_likelihood_hm204.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_VECTOR_LIKELIHOOD_PLAN_HM204.md",
    "grain_research/construction/math_vector_likelihood.py",
    "scripts/score_grain_vector_likelihood_cold.py",
    "scripts/run_grain_vector_likelihood_hm204.py",
    "tests/test_grain_vector_likelihood.py",
    "grain_research/construction/math_same_event_multitarget.py",
    "scripts/run_grain_observed_family_hm170.py",
)
DEFAULT = ROOT / "results/grain_math_response_v1/vector_likelihood_compiler_r1"
GROUPS = ("author_mean", "view0", "view1", "discovery", "held_author")
NATIVE_UNITS = ("score", "log_odds", "geometric_logit")


def arrays(path, keys=None):
    """Read only named numerical observations, without pickle or raw text."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def bind_parent(parent, stages):
    """Bind immutable source artifacts without opening historical private data."""
    inputs = {}
    for stage in stages:
        for name, expected in R.read(parent / stage).items():
            if R.digest(parent / name) != expected:
                raise ValueError("Changed parent artifact " + name)
        inputs[str((parent / stage).relative_to(ROOT))] = R.digest(parent / stage)
    return inputs


def number(x):
    """Undefined correlations are absent, not successful zeros."""
    return float(x) if np.isfinite(x) else None


def action(values):
    """Use remove-selected minus remove-control and selected minus control."""
    a = np.asarray(values, float)
    if a.ndim != 6 or a.shape[3:] != (5, 2, 5):
        raise ValueError(
            "Complete author/representation/view/arm/source/target family required"
        )
    return np.stack(
        (a[:, :, :, 1] - a[:, :, :, 2], a[:, :, :, 3] - a[:, :, :, 4]), axis=3
    )


def group_values(source, native, splits, group):
    """Keep authors as independent units; never pool their views as people."""
    if group.startswith("view"):
        view = int(group[-1])
        return source[:, view], native[:, view]
    mask = np.ones(len(splits), bool) if group == "author_mean" else splits == group
    if mask.sum() < 3:
        raise ValueError("Insufficient declared author group")
    return source[mask].mean(1), native[mask].mean(1)


def prepare(out):
    """Freeze source/application identities and numerical method before scoring."""
    cfg = R.read(ROOT / CONFIG)
    parent, native = ROOT / cfg["parent"], ROOT / cfg["native_parent"]
    R.check(
        parent,
        (
            "compiled_seal.json",
            "scored_seal.json",
            "analysis_seal.json",
            "verification_seal.json",
        ),
    )
    inputs = bind_parent(
        parent,
        (
            "prepared_seal.json",
            "compiled_seal.json",
            "scored_seal.json",
            "analysis_seal.json",
            "verification_seal.json",
        ),
    )
    inputs.update(
        bind_parent(
            native,
            (
                "prepared_seal.json",
                "collected_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
    )
    for p in (
        parent / "observations.npz",
        native / "00/observations.npz",
        native / "analysis.npz",
    ):
        inputs[str(p.relative_to(ROOT))] = R.digest(p)
    z = R.prepared(parent)
    n = arrays(
        native / "00/observations.npz",
        ("panel_vectors", "selected", "control", "splits"),
    )
    if any(
        not np.array_equal(z[k], n[k]) for k in ("selected", "control")
    ) or not np.array_equal(z["panel_vectors"][:, 0], n["panel_vectors"]):
        raise ValueError("Native comparator does not describe the same evidence")
    if z["source_vectors"].shape != (386, 2, 12, 1024) or z["panel_vectors"].shape != (
        24,
        2,
        2,
        12,
        1024,
    ):
        raise ValueError("Changed declared vector support")
    if set(z["source_indices"]) & set(z["panel_indices"]):
        raise ValueError("Source/application author overlap")
    points = np.empty((24, 2, 2, 5, 1024))
    records = []
    for u, rep, view in np.ndindex(24, 2, 2):
        ss, cc = z["selected"][u, view], z["control"][u, view]
        if len(set(ss) | set(cc)) != 8:
            raise ValueError("Duplicated or overlapping evidence selections")
        whole = np.arange(12)
        groups = (whole, whole[~np.isin(whole, ss)], whole[~np.isin(whole, cc)], ss, cc)
        for arm, slots in enumerate(groups):
            row = {
                "frame_key": f"{rep}/{u}/{view}/{arm}",
                "events": [
                    {
                        "event_id": f"{u}:{view}:{int(t)}",
                        "vector": z["panel_vectors"][u, rep, view, t].tolist(),
                    }
                    for t in slots
                ],
            }
            points[u, rep, view, arm] = M.cold_point(row, 1024)
            records.append(row)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source_vectors=z["source_vectors"],
        labels=z["labels"],
        source_indices=z["source_indices"],
        panel_indices=z["panel_indices"],
        points=points,
        splits=n["splits"],
    )
    with (out / "cold_input.jsonl").open("x") as sink:
        for row in records:
            sink.write(json.dumps(row, allow_nan=False) + "\n")
    R.write(
        out / "quality.json",
        {
            "source_authors": 386,
            "application_authors": 24,
            "source_application_overlap": 0,
            "cold_frames": len(records),
            "representation_names": cfg["representation_names"],
            "application_labels_read": 0,
            "new_model_calls": 0,
            "historical_selection": "N-enriched four-event selection retained;not new random sampling",
            "representation_overlap": "historical GRAIN/E5 development overlap retained;not lockbox",
        },
    )
    with (out / "data_schema.md").open("x") as sink:
        sink.write(
            "# HM204 numerical source and application evidence\n\n386 SOURCE authors,2 windows,12 normalized E5 comment vectors/window,1024D.\nFive inherited binary SOURCE labels in E/N/A/C/O order;no application labels.\n24 disjoint application authors,2 vector representations,2 windows,5 subsets.\n480 addressed subset means;windows are not independent participants.\nNative comparison is cue-mask only on exactly the same evidence IDs.\nNo timestamps/raw text/new encoder calls;means do not retain complete paths.\n"
        )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "quality.json", "data_schema.md"),
    )
    print(json.dumps(R.read(out / "quality.json")))


def compile_source(out):
    """Build a positive joint-profile law using source observations alone."""
    R.check(out)
    z = R.prepared(out)
    prototypes = z["source_vectors"].mean(2).transpose(1, 0, 2)
    variance = M.source_variance(prototypes.mean(0))
    labels = z["labels"]
    for bank in prototypes:
        M.validate(bank, labels, variance)
    cfg = R.read(out / "manifest.json")["config"]
    permutation = np.random.default_rng(cfg["null_permutation_seed"]).permutation(
        len(labels)
    )
    R.archive(
        out / "source_kernel.npz",
        prototypes=prototypes,
        labels=labels,
        permutation=permutation,
    )
    bundle = {
        "version": M.VERSION,
        "targets": list(M.TARGETS),
        "kernel": "source_kernel.npz",
        "kernel_sha256": R.digest(out / "source_kernel.npz"),
        "code_sha256": R.digest(Path(M.__file__)),
        "variance": variance,
        "reference": labels.mean(0).tolist(),
        "interpretation": "posterior_under_declared_source_Gaussian_law_not_human_probability",
    }
    R.write(out / "source_bundle.json", bundle)
    profile = (labels @ (2 ** np.arange(5))).astype(int)
    R.write(
        out / "source_reference.json",
        {
            "authors": len(labels),
            "binary_label_priors": labels.mean(0).tolist(),
            "joint_profile_counts": np.bincount(profile, minlength=32).tolist(),
            "label_covariance": np.cov(labels.T).tolist(),
            "variance": variance,
            "variance_units": "squared normalized-comment-mean Euclidean distance",
            "scope": "Gaussian channel declared,not empirically identified natural-language error",
        },
    )
    R.seal(
        out,
        "compiled_seal.json",
        ("source_kernel.npz", "source_bundle.json", "source_reference.json"),
    )
    print(
        json.dumps(
            {
                "variance": variance,
                "source_priors": bundle["reference"],
                "new_model_calls": 0,
            }
        )
    )


def score(out):
    """Invoke a separate cold process receiving no application labels or LLM answers."""
    R.check(out, ("compiled_seal.json",))
    command = [
        sys.executable,
        str(ROOT / "scripts/score_grain_vector_likelihood_cold.py"),
        "--bundle",
        str(out / "source_bundle.json"),
        "--expected-sha256",
        R.digest(out / "source_bundle.json"),
        "--input",
        str(out / "cold_input.jsonl"),
        "--output",
        str(out / "cold_output.jsonl"),
    ]
    R.write(out / "cold_invocation.json", {"command": command, "new_model_calls": 0})
    start = time.perf_counter()
    p = subprocess.run(command, capture_output=True, text=True, check=False)
    R.write(
        out / "cold_receipt.json",
        {
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "seconds": time.perf_counter() - start,
        },
    )
    if p.returncode:
        raise RuntimeError("Cold reader failed;retain invocation,do not retry")
    R.seal(
        out,
        "scored_seal.json",
        ("cold_invocation.json", "cold_receipt.json", "cold_output.jsonl"),
    )
    print(p.stdout.strip())


def analyze(out):
    """Compare local source-target fields and all native contacts without selection."""
    manifest = R.check(out, ("compiled_seal.json", "scored_seal.json"))
    z, kernel = R.prepared(out), arrays(out / "source_kernel.npz")
    b = R.read(out / "source_bundle.json")
    points = z["points"].reshape(-1, 1024)
    banks, targets = kernel["prototypes"], kernel["labels"]
    laws = [M.read(bank, targets, b["variance"], points) for bank in banks]
    nulls = [
        M.read(bank, targets[kernel["permutation"]], b["variance"], points)
        for bank in banks
    ]
    results = {
        key: np.stack([law[key] for law in laws], 1).reshape(24, 2, 2, 5, 2, 5)
        for key in ("posterior", "relative", "log_odds")
    }
    results["permuted_posterior"] = np.stack(
        [law["posterior"] for law in nulls], 1
    ).reshape(24, 2, 2, 5, 2, 5)
    global_fields = np.stack(
        [
            np.stack(
                [
                    (
                        bank[targets[:, q] == 1].mean(0)
                        - bank[targets[:, q] == 0].mean(0)
                    )
                    / b["variance"]
                    for q in range(5)
                ]
            )
            for bank in banks
        ]
    )
    global_values = np.einsum("avwhd,bqd->avwhbq", z["points"], global_fields)
    results["unconditional_contrast"] = global_values
    actions = {name: action(a) for name, a in results.items() if name != "relative"}
    native = arrays(
        ROOT / manifest["config"]["native_parent"] / "analysis.npz", ("native_actions",)
    )["native_actions"]
    if native.shape != (24, 2, 2, 5, 4):
        raise ValueError("Changed native action scope")
    rows, local_rows, field_summaries = [], [], []
    for i, (u, rep, view, arm) in enumerate(np.ndindex(24, 2, 2, 5)):
        for bank in range(2):
            for q, target in enumerate(M.TARGETS):
                rows.append(
                    {
                        "author": u,
                        "representation": manifest["config"]["representation_names"][
                            rep
                        ],
                        "view": view,
                        "arm": manifest["config"]["arms"][arm],
                        "source_view": bank,
                        "target": target,
                        "posterior": laws[bank]["posterior"][i, q],
                        "relative": laws[bank]["relative"][i, q],
                        "log_odds": laws[bank]["log_odds"][i, q],
                        "effective_prototypes": laws[bank]["effective_prototypes"][i],
                        "minimum_distance": laws[bank]["minimum_distance"][i],
                    }
                )
                field = laws[bank]["target_log_odds_field"][i, q]
                g = global_fields[bank, q]
                den = np.linalg.norm(field) * np.linalg.norm(g)
                local_rows.append(
                    {
                        "author": u,
                        "representation": manifest["config"]["representation_names"][
                            rep
                        ],
                        "view": view,
                        "arm": manifest["config"]["arms"][arm],
                        "source_view": bank,
                        "target": target,
                        "local_norm": np.linalg.norm(field),
                        "global_norm": np.linalg.norm(g),
                        "cosine": number(field @ g / den) if den else None,
                        "relative_field_difference": np.linalg.norm(field - g)
                        / np.linalg.norm(g),
                    }
                )
    for bank, law in enumerate(laws):
        for q, target in enumerate(M.TARGETS):
            f, g = law["target_log_odds_field"][:, q], global_fields[bank, q]
            cos = f @ g / (np.linalg.norm(f, axis=1) * np.linalg.norm(g))
            delta = (
                actions["log_odds"][:, :, :, :, bank, q]
                - actions["unconditional_contrast"][:, :, :, :, bank, q]
            )
            base = actions["log_odds"][:, :, :, :, bank, q]
            field_summaries.append(
                {
                    "source_view": bank,
                    "target": target,
                    "direction_cosine_min": float(cos.min()),
                    "direction_cosine_mean": float(cos.mean()),
                    "relative_field_difference_RMS": float(
                        np.sqrt(np.mean(np.sum((f - g) ** 2, axis=1)))
                        / np.linalg.norm(g)
                    ),
                    "finite_action_difference_RMS": float(np.sqrt(np.mean(delta**2))),
                    "finite_action_difference_over_local_RMS": float(
                        np.sqrt(np.mean(delta**2)) / np.sqrt(np.mean(base**2))
                    ),
                }
            )
    contacts = []
    for bank, op, group, reader, unit in np.ndindex(2, 2, len(GROUPS), len(actions), 3):
        name = list(actions)[reader]
        x, y = group_values(
            actions[name][:, 0, :, op, bank],
            native[:, :, op, :, unit],
            z["splits"],
            GROUPS[group],
        )
        r, cov = C.correlations(x, y), C.covariances(x, y)
        for q, p in np.ndindex(5, 5):
            contacts.append(
                {
                    "source_view": bank,
                    "operation": ("deletion", "isolated")[op],
                    "group": GROUPS[group],
                    "reader": name,
                    "native_unit": NATIVE_UNITS[unit],
                    "source_target": M.TARGETS[q],
                    "native_target": M.TARGETS[p],
                    "authors": len(x),
                    "r": number(r[q, p]),
                    "covariance": cov[q, p],
                }
            )
    summaries = []
    for bank, law, null in zip(range(2), laws, nulls, strict=True):
        summaries.append(
            {
                "source_view": bank,
                "effective_prototypes_range": [
                    float(law["effective_prototypes"].min()),
                    float(law["effective_prototypes"].max()),
                ],
                "posterior_ranges_by_target": np.column_stack(
                    [law["posterior"].min(0), law["posterior"].max(0)]
                ).tolist(),
                "relative_reading_RMS_by_target": np.sqrt(
                    np.mean(law["relative"] ** 2, axis=0)
                ).tolist(),
                "permuted_relative_RMS_by_target": np.sqrt(
                    np.mean(null["relative"] ** 2, axis=0)
                ).tolist(),
                "sigmoid_error": law["sigmoid_error"],
            }
        )
    matched = [
        r
        for r in contacts
        if r["reader"] in ("posterior", "log_odds")
        and r["source_target"] == r["native_target"]
        and r["group"] == "author_mean"
        and r["operation"] == "deletion"
        and r["native_unit"] == "score"
    ]
    R.table(out / "readings.csv", rows)
    R.table(out / "local_fields.csv", local_rows)
    R.table(out / "all_native_contacts.csv", contacts)
    R.archive(
        out / "analysis.npz",
        **results,
        **{"action_" + k: v for k, v in actions.items()},
        global_fields=global_fields,
        joint_profiles=np.stack([r["profile"] for r in laws], 1),
        native_actions=native,
    )
    R.write(
        out / "summary.json",
        {
            "packet": "HM204",
            "targets": list(M.TARGETS),
            "source_variance": b["variance"],
            "field_summaries": field_summaries,
            "source_readings": summaries,
            "matched_native_contacts": matched,
            "contacts": len(contacts),
            "new_model_calls": 0,
            "application_labels_read": 0,
            "answer_fit": False,
            "parameter_search": False,
            "construction": "positive SOURCE joint-profile posterior plus local all-target gradients",
            "scientific_status": "Constructive interface,not new Gaussian theorem or calibrated human prediction",
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analysis_seal.json",
        (
            "readings.csv",
            "local_fields.csv",
            "all_native_contacts.csv",
            "analysis.npz",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")))


def verify(out):
    """Check literal mixture, joint marginals, cold output and signed contacts once."""
    R.check(out, ("compiled_seal.json", "scored_seal.json", "analysis_seal.json"))
    z, kernel, saved = (
        R.prepared(out),
        arrays(out / "source_kernel.npz"),
        arrays(out / "analysis.npz"),
    )
    b = R.read(out / "source_bundle.json")
    points = z["points"].reshape(-1, 1024)
    literal_error = 0.0
    for bank, i in np.ndindex(2, 16):
        index = int(np.linspace(0, len(points) - 1, 16, dtype=int)[i])
        expected = M.literal(
            kernel["prototypes"][bank], kernel["labels"], b["variance"], points[index]
        )
        actual = saved["posterior"].reshape(-1, 2, 5)[index, bank]
        literal_error = max(literal_error, float(np.max(np.abs(actual - expected))))
    with (out / "cold_output.jsonl").open() as stream:
        cold = [json.loads(line) for line in stream]
    cold_error = 0.0
    seen = set()
    for r in cold:
        rep, u, view, arm = map(int, r["frame_key"].split("/"))
        bank = r["source_view"]
        key = (u, rep, view, arm, bank)
        if key in seen:
            raise ValueError("Cold output repeats evidence")
        seen.add(key)
        for k in ("posterior", "relative", "log_odds"):
            cold_error = max(
                cold_error, float(np.max(np.abs(np.asarray(r[k]) - saved[k][key])))
            )
        flat = np.ravel_multi_index(key[:4], (24, 2, 2, 5))
        cold_error = max(
            cold_error,
            float(
                np.max(
                    np.abs(
                        np.asarray(r["profile"]) - saved["joint_profiles"][flat, bank]
                    )
                )
            ),
        )
    bits = (np.arange(32)[:, None] // (2 ** np.arange(5))) % 2
    profiles = saved["joint_profiles"]
    marginal_error = float(
        np.max(np.abs(profiles @ bits - saved["posterior"].reshape(-1, 2, 5)))
    )
    normalization_error = float(np.max(np.abs(profiles.sum(-1) - 1)))
    import csv

    contact_error = 0.0
    checks = 0
    with (out / "all_native_contacts.csv").open() as stream:
        for row in csv.DictReader(stream):
            bank, op = (
                int(row["source_view"]),
                ("deletion", "isolated").index(row["operation"]),
            )
            unit = NATIVE_UNITS.index(row["native_unit"])
            x, y = group_values(
                saved["action_" + row["reader"]][:, 0, :, op, bank],
                saved["native_actions"][:, :, op, :, unit],
                z["splits"],
                row["group"],
            )
            a, d = (
                x[:, M.TARGETS.index(row["source_target"])],
                y[:, M.TARGETS.index(row["native_target"])],
            )
            aa, dd = a - a.mean(), d - d.mean()
            cov = sum(float(v * w) for v, w in zip(aa, dd, strict=True)) / (len(a) - 1)
            contact_error = max(contact_error, abs(cov - float(row["covariance"])))
            if row["r"]:
                rr = cov / (np.linalg.norm(aa) * np.linalg.norm(dd) / (len(a) - 1))
                contact_error = max(contact_error, abs(rr - float(row["r"])))
            checks += 1
    result = {
        "PASS": max(
            literal_error,
            cold_error,
            marginal_error,
            normalization_error,
            contact_error,
        )
        < 1e-10
        and len(seen) == 960,
        "literal_posterior_error": literal_error,
        "cold_replay_error": cold_error,
        "joint_marginal_error": marginal_error,
        "normalization_error": normalization_error,
        "contact_error": contact_error,
        "literal_points": 32,
        "cold_frames": len(seen),
        "contacts_replayed": checks,
        "new_model_calls": 0,
        "same_agent_not_independent": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result))
    if not result["PASS"]:
        raise RuntimeError(
            "Finite accounts failed;preserve failure,do not optimize precision"
        )


def main():
    """Run each bounded scientific stage once in an exclusive result directory."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=("prepare", "compile", "score", "analyze", "verify")
    )
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    start = time.perf_counter()
    {
        "prepare": prepare,
        "compile": compile_source,
        "score": score,
        "analyze": analyze,
        "verify": verify,
    }[args.stage](args.out)
    print(json.dumps({"stage": args.stage, "seconds": time.perf_counter() - start}))


if __name__ == "__main__":
    main()

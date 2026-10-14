#!/usr/bin/env python3
"""Run a bounded, numeric-only PANDORA cross-cohort configuration experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_configuration_kernel as K

SPEC = "configs/grain_configuration_kernel_hm144.json"
MATERIAL = "results/grain_vector_mechanisms/predecessor_integration_v1/material/"
INPUTS = {
    "manifest": MATERIAL + "material_manifest.json",
    "vectors": MATERIAL + "private_arrays.npz",
    "addresses": "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv",
    "roles": "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json",
    "directions": "results/grain_construction_v1/C3/anonymous_evidence_r1/private_learned_directions.npz",
}
CODE = [
    SPEC,
    "docs/GRAIN_CONFIGURATION_KERNEL_PLAN_HM144.md",
    "grain_research/construction/math_configuration_kernel.py",
    "scripts/run_grain_configuration_kernel_hm144.py",
    "tests/test_grain_configuration_kernel.py",
]
OUTPUT = ROOT / "results/grain_math_response_v1/configuration_kernel_r1"


def read(path):
    """Read a non-text scientific JSON artifact."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Stream a file digest without interpreting other NPZ arrays or raw text."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Save once; never overwrite a frozen scientific artifact."""
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def table(path, rows):
    """Write a complete aggregate table with explicit fields."""
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def archive(path, arrays):
    """Save local sensitive numerical derivatives; no raw text or scale labels."""
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def prepare(out):
    """Seal the finite scope and existing inputs before outcome computation."""
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "contract.json",
        {
            "spec": read(ROOT / SPEC),
            "inputs": {
                k: {"path": p, "sha256": sha(ROOT / p)} for k, p in INPUTS.items()
            },
            "code": {p: sha(ROOT / p) for p in CODE},
            "reads": "cached comments array and selected numeric/event/venue metadata only",
            "forbidden": "raw texts, labels, item wording, MEPS, API, training, teacher fit",
            "scope_complete": False,
        },
    )


def verify_contract(out):
    """Check source and code identity before using registered material."""
    c = read(out / "contract.json")
    for item in c["inputs"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("changed registered input")
    for p, digest in c["code"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("changed registered code or plan")
    return c["spec"]


def load_numeric():
    """Load only the authorized comments key, anonymous directions and addresses."""
    manifest = read(ROOT / INPUTS["manifest"])
    if sha(ROOT / INPUTS["vectors"]) != manifest["files"]["private_arrays.npz"]:
        raise ValueError("vector provenance mismatch")
    with np.load(ROOT / INPUTS["vectors"]) as f:
        vectors = f["comments"].astype(float)
    if vectors.shape != (624, 36, 1024) or not np.isfinite(vectors).all():
        raise ValueError("unexpected comment observation shape")
    a = (
        pd.read_csv(
            ROOT / INPUTS["addresses"],
            usecols=[
                "author_index",
                "unit",
                "start",
                "end",
                "condition",
                "timestamp_unix_seconds",
                "body_sha256",
            ],
        )
        .sort_values(["author_index", "unit"])
        .reset_index(drop=True)
    )
    if len(a) != 624 * 36 or a.duplicated(["author_index", "unit"]).any():
        raise ValueError("incomplete or duplicate vector event address")
    if a.body_sha256.duplicated().any():
        raise ValueError("repeated publication text needs a separate observation model")
    if not np.array_equal(a.author_index, np.repeat(np.arange(624), 36)):
        raise ValueError("author-to-vector alignment mismatch")
    if not np.array_equal(a.unit, np.tile(np.arange(36), 624)):
        raise ValueError("event-to-vector alignment mismatch")
    venue = a.condition.to_numpy().reshape(624, 36)
    times = a.timestamp_unix_seconds.to_numpy(float).reshape(624, 36)
    lengths = (a.end - a.start).to_numpy(float).reshape(624, 36)
    if not np.isfinite(times).all() or (lengths <= 0).any():
        raise ValueError("invalid observed time or span length")
    roles = {
        k: np.asarray(v, dtype=int) for k, v in read(ROOT / INPUTS["roles"]).items()
    }
    combined = np.concatenate(list(roles.values()))
    if len(np.unique(combined)) != 624 or not np.array_equal(
        np.sort(combined), np.arange(624)
    ):
        raise ValueError("source and application authors overlap or are missing")
    with np.load(ROOT / INPUTS["directions"]) as f:
        direction = np.column_stack(
            [
                f[k]
                for k in (
                    "author_stability",
                    "response_stability",
                    "principal_components",
                )
            ]
        )
    norm = np.linalg.norm(vectors, axis=2, keepdims=True)
    if (norm == 0).any():
        raise ValueError("zero encoder vectors")
    return vectors / norm, direction, venue, times, lengths, roles


def metadata_support(venue, lengths, roles):
    """Keep coarse author-chosen conditions; inspect exact support without names."""
    src = roles["source_train"]
    counts = Counter(venue[src, :25].ravel().tolist())
    top = sorted(counts, key=lambda s: (-counts[s], s))[:20]
    choice = np.isin(venue, top).mean(1)
    span = np.log1p(lengths).mean(1)
    cuts = [float(np.median(choice[src])), float(np.median(span[src]))]
    strata = 2 * (choice >= cuts[0]).astype(int) + (span >= cuts[1]).astype(int)
    authors = [{v for v in row} for row in venue]
    candidates = []
    for triple in combinations(range(20), 3):
        selected = {top[j] for j in triple}
        candidates.append((sum(selected <= authors[u] for u in src), triple))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    best = {top[j] for j in candidates[0][1]}
    return strata, {
        "role_counts": {k: len(v) for k, v in roles.items()},
        "unique_venues": len(set(venue.ravel())),
        "events": int(venue.size),
        "best_source_chosen_exact_venue_triple_support": {
            k: sum(best <= authors[u] for u in us) for k, us in roles.items()
        },
        "exact_triple_not_used": True,
        "strata_source_cuts": cuts,
        "strata_counts": {
            k: np.bincount(strata[us], minlength=4).tolist() for k, us in roles.items()
        },
        "matching_is_coarse_not_full_topic_control": True,
        "no_condition_residualization": True,
    }


def comparison(source, application):
    """Construct fixed-kernel joint and pair signed comparisons, not a predictor."""
    widths = [K.source_bandwidth(source[:, j]) for j in range(3)]
    ss = [K.rbf(source[:, j], source[:, j], widths[j]) for j in range(3)]
    aa = [K.rbf(application[:, j], application[:, j], widths[j]) for j in range(3)]
    sa = [K.rbf(source[:, j], application[:, j], widths[j]) for j in range(3)]
    s_norm, a_norm, action = (
        K.interaction_inner(ss),
        K.interaction_inner(aa),
        K.interaction_inner(sa),
    )
    pairs = {}
    for i, j in combinations(range(3), 2):
        inner = K.interaction_inner([sa[i], sa[j]])
        sn = K.interaction_inner([ss[i], ss[j]])
        an = K.interaction_inner([aa[i], aa[j]])
        pairs[f"{i}{j}"] = {"inner": inner, "alignment": inner / np.sqrt(sn * an)}
    result = {
        "source_lancaster_norm_squared": s_norm,
        "application_lancaster_norm_squared": a_norm,
        "signed_lancaster_inner": action,
        "signed_alignment_not_pearson_r": action / np.sqrt(s_norm * a_norm),
        "source_dHSIC": K.total_independence(ss),
        "application_dHSIC": K.total_independence(aa),
        "pairs": pairs,
        "bandwidth_squared": widths,
    }
    return result, ss, aa, sa


def run(out):
    """Compute all registered channels, partitions and recipient cohorts once."""
    spec = verify_contract(out)
    write(out / "execution_started.json", {"packet": "HM-144", "status": "RUNNING"})
    began = time.monotonic()
    vectors, direction, venue, times, lengths, roles = load_numeric()
    strata, support = metadata_support(venue, lengths, roles)
    chronological = np.argsort(times, axis=1, kind="stable").reshape(624, 3, 12)
    balanced = np.stack([K.balanced_views(venue[u], times[u]) for u in range(624)])
    partitions = {"chronological": chronological, "venue_balanced": balanced}
    grain = vectors @ direction
    src = roles["source_train"]
    mean, sd = grain[src].mean((0, 1)), grain[src].std((0, 1))
    if (sd <= 0).any():
        raise ValueError("source feature unit degenerate")
    spaces = {"grain15": (grain - mean) / sd, "e5_1024": vectors}
    saved = {"grain_source_mean": mean, "grain_source_sd": sd, "strata": strata}
    saved.update({f"role_{k}": v for k, v in roles.items()})
    saved.update({f"partition_{k}": v for k, v in partitions.items()})
    rows, result_rows = [], []
    for partition, indices in partitions.items():
        for space, values in spaces.items():
            parcels = np.take_along_axis(
                values[:, None], indices[..., None], axis=2
            ).mean(2)
            saved[f"parcels_{partition}_{space}"] = parcels
            for recipient in spec["recipients"]:
                us = roles[recipient]
                label = f"{partition}__{space}__{recipient}"
                obs, ss, aa, sa = comparison(parcels[src], parcels[us])
                saved[f"source_witness_{label}"] = K.source_witness(ss, sa)
                for name, mats in (("ss", ss), ("aa", aa), ("sa", sa)):
                    for j, mat in enumerate(mats):
                        saved[f"{name}_{j}_{label}"] = mat
                rng = np.random.default_rng(spec["seed"])
                null_summary = {}
                for matching in ("global", "coarse_condition"):
                    group = None if matching == "global" else strata[us]
                    orders = np.stack(
                        [
                            K.permutation(len(us), rng, group)
                            for _ in range(spec["permutations"])
                        ]
                    )
                    saved[f"orders_{matching}_{label}"] = orders
                    values_null = K.cut_readings(sa, orders)
                    saved[f"null_{matching}_{label}"] = values_null
                    tails = (
                        (values_null >= obs["signed_lancaster_inner"]).sum(0) + 1
                    ) / (len(values_null) + 1)
                    null_summary[matching] = {
                        "cut_tail_fractions": tails.tolist(),
                        "max_cut_tail_fraction": float(tails.max()),
                        "cut_mean": values_null.mean(0).tolist(),
                        "cut_95_quantiles": np.quantile(
                            values_null, [0.025, 0.975], axis=0
                        ).tolist(),
                    }
                boot = K.bootstrap_inner(sa, spec["bootstrap_draws"], spec["seed"])
                saved[f"bootstrap_{label}"] = boot
                ci = np.quantile(boot, [0.025, 0.975]).tolist()
                item = {
                    "partition": partition,
                    "space": space,
                    "recipient": recipient,
                    **obs,
                    "signed_inner_bootstrap_95CI_frozen_features": ci,
                    "relinks": null_summary,
                }
                result_rows.append(item)
                rows.append(
                    {
                        "partition": partition,
                        "space": space,
                        "recipient": recipient,
                        "source_n": len(src),
                        "recipient_n": len(us),
                        "signed_inner": obs["signed_lancaster_inner"],
                        "alignment_not_personality_r": obs[
                            "signed_alignment_not_pearson_r"
                        ],
                        "CI_low": ci[0],
                        "CI_high": ci[1],
                        "global_max_cut_tail": null_summary["global"][
                            "max_cut_tail_fraction"
                        ],
                        "coarse_max_cut_tail": null_summary["coarse_condition"][
                            "max_cut_tail_fraction"
                        ],
                        "pair01_alignment": obs["pairs"]["01"]["alignment"],
                        "pair02_alignment": obs["pairs"]["02"]["alignment"],
                        "pair12_alignment": obs["pairs"]["12"]["alignment"],
                    }
                )
    archive(out / "private_kernel_arrays.npz", saved)
    table(out / "metrics.csv", rows)
    write(out / "support.json", support)
    write(
        out / "result.json",
        {
            "packet": "HM-144",
            "status": "NUMERICAL_CONFIGURATION_QUESTION_EXECUTED_NOT_PSYCHOMETRIC_CLOSURE",
            "comparisons": result_rows,
            "CPU_wall_seconds": time.monotonic() - began,
            "new_model_API_private_raw_training_calls": 0,
            "raw_text_or_label_keys_read": [],
            "original_scope_complete": False,
            "scope": "cross-material author configuration, not causal personality, unknown local classes, or all-pair-irreducible synergy",
        },
    )
    write(
        out / "output_seal.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    )
    print(
        json.dumps(
            {
                "status": "EXECUTED",
                "comparisons": len(rows),
                "seconds": time.monotonic() - began,
            }
        )
    )


def verify(out):
    """Independently reconstruct empirical tensors using centering matrix products."""
    verify_contract(out)
    for name, digest in read(out / "output_seal.json").items():
        if sha(out / name) != digest:
            raise ValueError("changed frozen output")
    results = read(out / "result.json")["comparisons"]
    errors = []
    with np.load(out / "private_kernel_arrays.npz") as a:
        for partition in ("chronological", "venue_balanced"):
            addresses = a[f"partition_{partition}"]
            if not np.array_equal(
                np.sort(addresses.reshape(624, 36)), np.tile(np.arange(36), (624, 1))
            ):
                raise ValueError("parcels omit or reuse a publication event")
        for obs in results:
            label = f"{obs['partition']}__{obs['space']}__{obs['recipient']}"
            cross = [a[f"sa_{j}_{label}"] for j in range(3)]
            n, m = cross[0].shape
            hs, ha = np.eye(n) - np.ones((n, n)) / n, np.eye(m) - np.ones((m, m)) / m
            alternative = [hs @ g @ ha for g in cross]
            value = float(np.prod(alternative, axis=0).sum() / (n * m))
            errors.append(abs(value - obs["signed_lancaster_inner"]))
            for matching in ("global", "coarse_condition"):
                orders = a[f"orders_{matching}_{label}"]
                expected = a[f"null_{matching}_{label}"]
                for cut in range(3):
                    base = alternative[(cut + 1) % 3] * alternative[(cut + 2) % 3]
                    for j, order in enumerate(orders):
                        val = float(
                            np.einsum("ij,ij->", base, alternative[cut][:, order])
                            / (n * m)
                        )
                        errors.append(abs(val - expected[j, cut]))
                if matching == "coarse_condition":
                    groups = a["strata"][a[f"role_{obs['recipient']}"]]
                    if not np.all(groups[orders] == groups):
                        raise ValueError("coarse strata not preserved by relinking")
            boot = a[f"bootstrap_{label}"]
            errors.extend(
                abs(
                    np.quantile(boot, [0.025, 0.975])
                    - obs["signed_inner_bootstrap_95CI_frozen_features"]
                )
            )
    if max(errors) > 1e-10:
        raise ValueError("alternative arithmetic failed")
    print(
        json.dumps(
            {
                "status": "READONLY_ALTERNATE_CHECK_PASS",
                "comparisons": len(results),
                "arithmetic_checks": len(errors),
                "maximum_error": max(errors),
            }
        )
    )


def main():
    """Expose distinct sealing, execution and repeatable read-only verification."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run", "verify"])
    p.add_argument("--out", type=Path, default=OUTPUT)
    a = p.parse_args()
    {"prepare": prepare, "run": run, "verify": verify}[a.command](a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Execute a fixed author-aggregation mechanism packet on sealed numeric carriers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from functools import reduce
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_evidence_merge as M

SPEC = ROOT / "configs/grain_author_merge_hm102.json"
CODE = (
    "configs/grain_author_merge_hm102.json",
    "docs/GRAIN_AUTHOR_MERGE_PLAN_HM102.md",
    "grain_research/construction/math_author_evidence_merge.py",
    "grain_research/construction/math_cold_evidence.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/score_grain_author_evidence_merge_cold.py",
    "tests/test_grain_math_author_evidence_merge.py",
)


def load(path):
    """Load one explicitly addressed JSON artifact."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash exact substantive input bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Never replace existing or failed research artifacts."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_hashes(base, hashes):
    """Verify all admitted source artifacts before reading their scientific values."""
    for name, digest in hashes.items():
        if sha(base / name) != digest:
            raise ValueError("Sealed artifact changed: " + name)


def seals(base, folder):
    """Ignore only macOS resource forks, never substantive failed outputs."""
    return {
        str(p.relative_to(base)): sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.name.startswith("._")
    }


def table(path, rows):
    """Write every declared signed comparison, with no winner filtering."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check(out):
    """Authenticate fixed code, source lineage and the cold numeric interface."""
    manifest = load(out / "00/manifest.json")
    verify_hashes(ROOT, manifest["code"] | manifest["source"])
    verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out, spec):
    """Compile the query-only witness before exporting any application K/V."""
    if out.exists():
        raise FileExistsError("Preserve prior and partial packet")
    source = ROOT / spec["source"]
    old = load(source / "00/manifest.json")
    verify_hashes(ROOT, old["code"] | old["source"])
    verify_hashes(source, old["inputs"])
    verify_hashes(source, load(source / "01/raw_seal.json"))
    verify_hashes(source, load(source / "02/output_seal.json"))
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM101_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM101 actual intervention captures required")
    rows, bundle = load(source / "00/rows.json"), load(source / "00/bundle.json")
    dest = out / "00"
    dest.mkdir(parents=True)
    positions = {}
    for endpoint in sorted({max(r["current"]) for r in rows}):
        q = M.registered_queries(bundle, endpoint)
        delta, diag = M.common_key_offset(
            q, spec["key_heads"], spec["query_lstsq_rcond"]
        )
        positions[str(endpoint)] = {
            "queries": q.tolist(),
            "key_offset": delta.tolist(),
            "diagnostics": diag,
        }
        if max(d["residual"] for d in diag) > spec["vector_witness_atol"]:
            write(
                out / "vector_witness_failure.json",
                {
                    "endpoint": endpoint,
                    "diagnostics": diag,
                    "status": "FAILED_NOT_REPAIRED",
                },
            )
            raise ValueError("Common all-target vector gauge not realizable")
    reader = {
        "targets": bundle["targets"],
        "directions": bundle["directions"],
        "positions": positions,
        "query_only_witness_no_predictor_fit": True,
    }
    write(dest / "reader.json", reader)
    keys, values, offsets, endpoints, indices = [], [], [0], [], []
    for row in rows:
        with np.load(
            source / f"01/units/unit_{row['unit']:04d}.npz", allow_pickle=False
        ) as z:
            keys.append(z["k"].astype(float))
            values.append(z["v"].astype(float))
        if len(keys[-1]) != len(row["current"]):
            raise ValueError("Carrier/export record boundary differs")
        offsets.append(offsets[-1] + len(keys[-1]))
        endpoints.append(max(row["current"]))
        indices.append(
            [
                row[n]
                for n in (
                    "corpus",
                    "bridge",
                    "family",
                    "previous",
                    "current_state",
                    "arm",
                )
            ]
        )
    with (dest / "carriers.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            keys=np.concatenate(keys),
            values=np.concatenate(values),
            offsets=offsets,
            endpoints=endpoints,
            indices=indices,
        )
    source_files = [
        source / n
        for n in (
            "00/manifest.json",
            "00/bundle.json",
            "00/rows.json",
            "01/raw_seal.json",
            "02/output_seal.json",
            "02/accounts.npz",
            "verification.json",
        )
    ]
    source_files.extend(source / n for n in load(source / "01/raw_seal.json"))
    write(
        dest / "manifest.json",
        {
            "spec": spec,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
            "inputs": seals(out, dest),
            "records": len(rows),
            "reader_compiled_before_carrier_export": True,
            "original_scope_complete": False,
        },
    )
    print("PREPARED HM102", len(rows), "records; query-only vector witness compiled")


def analyze(out):
    """Keep local invariance and all author-level normalization consequences."""
    manifest = check(out)
    spec, reader = manifest["spec"], load(out / "00/reader.json")
    with np.load(out / "00/carriers.npz", allow_pickle=False) as z:
        data = {n: z[n] for n in z.files}
    with np.load(out / "01_cold.npz", allow_pickle=False) as z:
        f, mass = z["readings"], z["log_mass"]
    if f.shape != (manifest["records"], 6, 12) or mass.shape != f.shape:
        raise ValueError("Incomplete cold score output")
    all_f, all_z = [], []
    local_error = shift_error = 0.0
    for name, amplitude in spec["gauges"].items():
        fg, zg = [], []
        for unit, (start, end, endpoint) in enumerate(
            zip(
                data["offsets"][:-1],
                data["offsets"][1:],
                data["endpoints"],
                strict=True,
            )
        ):
            pos = reader["positions"][str(int(endpoint))]
            theta = (
                amplitude
                if name == "global"
                else amplitude * (data["indices"][unit, 3] - 1)
            )
            value, logz = M.observe(
                data["keys"][start:end] + theta * np.asarray(pos["key_offset"]),
                data["values"][start:end],
                pos["queries"],
                reader["directions"],
            )
            local_error = max(local_error, float(np.max(np.abs(value - f[unit]))))
            shift_error = max(
                shift_error, float(np.max(np.abs(logz - mass[unit] - theta)))
            )
            fg.append(value)
            zg.append(logz)
        all_f.append(fg)
        all_z.append(zg)
    all_f, all_z = np.array(all_f), np.array(all_z)
    if max(local_error, shift_error) > spec["arithmetic_atol"]:
        write(
            out / "gauge_failure.json",
            {"local_error": local_error, "shift_error": shift_error},
        )
        raise ValueError("Vector-realized local gauge control failed")
    source = ROOT / spec["source"]
    with np.load(source / "02/accounts.npz", allow_pickle=False) as z:
        parent_fields, parent_contrasts = z["fields"], z["contrasts"]
    replay = max(
        abs(f[u].sum(-1) - parent_fields[tuple(ix)][1]).max()
        for u, ix in enumerate(data["indices"])
    )
    laws = M.author_laws()
    results, control, full_expected, full_merged = [], [], {}, {}
    covariance_error = expected_replay = expected_invariance = 0.0
    for bank, bridge, family, arm in itertools.product(
        range(2), range(2), range(6), range(4)
    ):
        select = np.flatnonzero(
            np.all(
                data["indices"][:, [0, 1, 2, 5]] == [bank, bridge, family, arm], axis=1
            )
        )
        select = select[
            np.lexsort((data["indices"][select, 4], data["indices"][select, 3]))
        ]
        if len(select) != 9 or not np.array_equal(
            data["indices"][select, 3:5],
            np.array(list(itertools.product(range(3), repeat=2))),
        ):
            raise ValueError("Incomplete paired author-law support")
        for atom in spec["atom_conventions"]:
            counts = np.diff(data["offsets"])[select, None, None]
            for gi, name in enumerate(spec["gauges"]):
                lf = all_f[gi, select]
                lz = all_z[gi, select] - (np.log(counts) if atom == "unit_mean" else 0)
                comparisons = [M.compose(lf, lz, p) for p in laws]
                for e, m, c in comparisons:
                    covariance_error = max(
                        covariance_error, float(np.max(np.abs(m - e - c)))
                    )
                expected, merged = (
                    comparisons[0][k].sum(-1) - comparisons[1][k].sum(-1)
                    for k in (0, 1)
                )
                expected_replay = max(
                    expected_replay,
                    float(
                        np.max(
                            np.abs(
                                expected
                                - parent_contrasts[bank, bridge, family, arm, 1]
                            )
                        )
                    ),
                )
                key = (bank, bridge, family, arm, atom)
                if gi == 0:
                    full_expected[key], full_merged[key] = expected, merged
                expected_invariance = max(
                    expected_invariance,
                    float(np.max(np.abs(expected - full_expected[key]))),
                )
                for t, target in enumerate(reader["targets"]):
                    record = {
                        "bank": bank,
                        "bridge": bridge,
                        "family": family,
                        "arm": arm,
                        "atom": atom,
                        "gauge": name,
                        "target": target,
                        "expected_contrast": float(expected[t]),
                        "partition_merge_contrast": float(merged[t]),
                        "normalization_gap": float(merged[t] - expected[t]),
                        "merge_change_from_base": float(
                            merged[t] - full_merged[key][t]
                        ),
                    }
                    results.append(record)
                    if arm == 0:
                        control.append(record)
    summary = {
        "status": "ANALYZED_FINITE_HM102_NOT_GLOBAL",
        "new_neural_calls": 0,
        "predictor_fits": 0,
        "records": len(f),
        "local_gauge_reading_error": local_error,
        "common_log_partition_shift_error": shift_error,
        "parent_local_replay_error": float(replay),
        "parent_author_replay_error": expected_replay,
        "expected_author_gauge_error": expected_invariance,
        "covariance_identity_error": covariance_error,
        "neither_expected_max": max(abs(r["expected_contrast"]) for r in control),
        "neither_gauged_merge_max": max(
            abs(r["partition_merge_contrast"])
            for r in control
            if r["gauge"].startswith("history")
        ),
        "neither_base_merge_max": max(
            abs(r["partition_merge_contrast"]) for r in control if r["gauge"] == "base"
        ),
        "merge_gauge_change_max": max(
            abs(r["merge_change_from_base"]) for r in results
        ),
        "global_gauge_change_max": max(
            abs(r["merge_change_from_base"]) for r in results if r["gauge"] == "global"
        ),
        "witness_residual_max": max(
            d["residual"]
            for pos in reader["positions"].values()
            for d in pos["diagnostics"]
        ),
        "witness_norm_max": max(
            d["norm"]
            for pos in reader["positions"].values()
            for d in pos["diagnostics"]
        ),
        "original_scope_complete": False,
        "psychological_accuracy_claim": False,
    }
    dest = out / "02"
    dest.mkdir()
    table(dest / "author_scores.csv", results)
    table(dest / "no_history_control.csv", control)
    with (dest / "gauged_observations.npz").open("xb") as stream:
        np.savez_compressed(stream, readings=all_f, log_mass=all_z)
    write(dest / "summary.json", summary)
    write(
        out / "output_seal.json",
        seals(out, dest) | {"01_cold.npz": sha(out / "01_cold.npz")},
    )
    if (
        max(replay, expected_replay, expected_invariance, covariance_error)
        > spec["arithmetic_atol"]
    ):
        raise ValueError("Frozen observation replay or author account failed")
    print("ANALYZED HM102", json.dumps(summary))


def audit(out):
    """Recompute direct token sums and stream laws, not merely file/test counts."""
    manifest = check(out)
    verify_hashes(out, load(out / "output_seal.json"))
    reader, spec = load(out / "00/reader.json"), manifest["spec"]
    with np.load(out / "00/carriers.npz", allow_pickle=False) as z:
        data = {n: z[n] for n in z.files}
    with np.load(out / "02/gauged_observations.npz", allow_pickle=False) as z:
        fields, partitions = z["readings"], z["log_mass"]
    rows = list(csv.DictReader((out / "02/author_scores.csv").open()))
    direct_error = stream_error = 0.0
    checks = 0
    for bank, bridge, family, arm in itertools.product(
        range(2), range(2), range(6), range(4)
    ):
        select = np.flatnonzero(
            np.all(
                data["indices"][:, [0, 1, 2, 5]] == [bank, bridge, family, arm], axis=1
            )
        )
        select = select[
            np.lexsort((data["indices"][select, 4], data["indices"][select, 3]))
        ]
        for atom, (gi, name) in itertools.product(
            spec["atom_conventions"], enumerate(spec["gauges"])
        ):
            scores = []
            for p in M.author_laws():
                num, den = [], []
                blocks = []
                for n, unit in enumerate(select):
                    start, end = data["offsets"][unit : unit + 2]
                    pos = reader["positions"][str(int(data["endpoints"][unit]))]
                    q, d = np.asarray(pos["queries"]), np.asarray(reader["directions"])
                    theta = (
                        spec["gauges"][name]
                        if name == "global"
                        else spec["gauges"][name] * (data["indices"][unit, 3] - 1)
                    )
                    kk = data["keys"][start:end] + theta * np.asarray(pos["key_offset"])
                    vv = data["values"][start:end]
                    logz = partitions[gi, unit] - (
                        np.log(end - start) if atom == "unit_mean" else 0
                    )
                    blocks.append((logz + np.log(p[n]), fields[gi, unit]))
                    head = np.arange(q.shape[1]) // (q.shape[1] // kk.shape[1])
                    # This independent direct token measure keeps each record's query position.
                    logits = np.einsum("ahd,thd->ath", kk[:, head], q)
                    content = np.einsum("ahd,thd->ath", vv[:, head], d)
                    token_log_mass = (
                        logits
                        + np.log(p[n])
                        - (np.log(end - start) if atom == "unit_mean" else 0)
                    )
                    den.append(token_log_mass)
                    num.append(content)
                ell, value = np.concatenate(den), np.concatenate(num)
                weight = np.exp(ell - ell.max(0))
                direct = (weight * value).sum(0) / weight.sum(0)
                stream = reduce(M.merge_lifts, blocks)
                reverse = reduce(M.merge_lifts, reversed(blocks))
                split = M.merge_lifts(
                    reduce(M.merge_lifts, blocks[:4]), reduce(M.merge_lifts, blocks[4:])
                )
                stream_error = max(
                    stream_error,
                    float(np.max(np.abs(stream[1] - direct))),
                    float(np.max(np.abs(reverse[1] - direct))),
                    float(np.max(np.abs(split[1] - direct))),
                )
                scores.append(direct.sum(-1))
                checks += direct.size * 3
            selected = [
                r
                for r in rows
                if (
                    int(r["bank"]),
                    int(r["bridge"]),
                    int(r["family"]),
                    int(r["arm"]),
                    r["atom"],
                    r["gauge"],
                )
                == (bank, bridge, family, arm, atom, name)
            ]
            target_values = {
                r["target"]: float(r["partition_merge_contrast"]) for r in selected
            }
            direct_error = max(
                direct_error,
                max(
                    abs(scores[0][t] - scores[1][t] - target_values[target])
                    for t, target in enumerate(reader["targets"])
                ),
            )
    if max(direct_error, stream_error) > spec["arithmetic_atol"]:
        write(
            out / "verification_failure.json",
            {"direct_error": direct_error, "stream_error": stream_error},
        )
        raise ValueError("Direct token / stream merge verification failed")
    report = {
        "status": "VERIFIED_FINITE_HM102_NOT_GLOBAL",
        "direct_token_contrast_error": direct_error,
        "stream_reorder_split_error": stream_error,
        "stream_scalar_checks": checks,
        "same_agent_not_third_party": True,
        "new_neural_calls": 0,
        "original_scope_complete": False,
    }
    write(out / "verification.json", report)
    print("VERIFIED HM102", json.dumps(report))


def main():
    """Explicit finite stages; no autonomous retries or hidden resource queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze", "verify"))
    args = parser.parse_args()
    spec = load(SPEC)
    out = ROOT / spec["output"]
    if args.stage == "prepare":
        prepare(out, spec)
    elif args.stage == "analyze":
        analyze(out)
    else:
        audit(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HM36 bounded CPU source-active support and matched-rank cold controls."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_mixed_query_path as M
from grain_research.construction import math_target_active_support as T
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_query_response import load_model
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_target_active_support_v1.json"
PLAN = ROOT / "docs/GRAIN_TARGET_ACTIVE_QUERY_SUPPORT_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/target_active_support_r1"
CODE = (
    "grain_research/construction/math_target_active_support.py",
    "scripts/run_grain_math_target_active_support.py",
    "scripts/verify_grain_math_target_active_support.py",
    "tests/test_grain_math_target_active_support.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def save(path, metadata, arrays):
    """Atomic uncompressed arrays; metadata commits last to avoid false completion."""
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path.with_suffix(".npz"))
    write_json(path, {**metadata, "array_sha256": digest(path.with_suffix(".npz"))})


def load(path):
    """Read one committed hash-checked array bundle."""
    meta = array_unit(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
        return meta, {k: a[k].copy() for k in a.files}


def source_node(parent, path, node, role):
    """Require the role/address before loading source or application numerical data."""
    file = parent / "01/nodes" / f"path_{path['path']:03d}_node_{node}.json"
    meta = array_unit(file)
    if path["role"] != role or meta["address"] != path or meta["node"] != node:
        raise ValueError("node role or address differs from frozen design")
    with np.load(file.with_suffix(".npz"), allow_pickle=False) as a:
        return (
            file,
            a["p_covectors"].astype(float),
            a["delta_p"].astype(float),
            a["supported"].astype(float),
        )


def prepare(out):
    """Freeze dependencies, roles and code without loading any numerical arrays."""
    if out.exists():
        raise FileExistsError("preserve prior packet; no automatic overwrite")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    pm = check(parent)
    verified = ROOT / spec["parent_verification"] / "summary.json"
    if read_json(verified)["status"] != "VERIFIED_FINITE_HM35_NOT_GLOBAL":
        raise ValueError("verified actual HM35 required")
    files = dict(pm["files"])
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("parent result changed")
        files[str((parent / name).relative_to(ROOT))] = sha
    files[str(verified.relative_to(ROOT))] = digest(verified)
    for name in CODE:
        files[name] = digest(ROOT / name)
    design = read_json(parent / "00/design.json")
    if len(design) != 96 or sum(p["role"] == "source" for p in design) != 48:
        raise ValueError("original full HM35 design required")
    for sub in ("00/code", "01/models", "02"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "frozen_at": now(),
            "new_model_calls": 0,
            "neural_training": False,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "learned_bases": 0,
            "random_bases": 0,
            "source_nodes_loaded": 0,
            "application_nodes_loaded": 0,
            "analysis_seconds": 0.0,
            "new_model_calls": 0,
        },
    )
    print({"status": "PREPARED", "paths": len(design)}, flush=True)


class Budget:
    """Cumulative bounded wall analysis time across explicit resumable stages."""

    def __init__(self, out, spec):
        self.out, self.spec = out, spec
        self.initial = read_json(out / "run_status.json")["analysis_seconds"]
        self.start = time.monotonic()

    def checkpoint(self, status):
        """Save actual elapsed time before enforcing the fixed ceiling."""
        status["analysis_seconds"] = self.initial + time.monotonic() - self.start
        write_json(self.out / "run_status.json", status)
        if status["analysis_seconds"] > self.spec["maximum_analysis_seconds"]:
            raise TimeoutError("fixed 900-second analysis budget reached")


def fit_source(out):
    """Nine source bases and eighteen fixed controls; no application arrays read."""
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    if status["status"] != "PREPARED":
        raise ValueError(
            "untouched source stage required; partial fits are not replayed"
        )
    budget = Budget(out, spec)
    status["status"] = "SOURCE_RUNNING"
    budget.checkpoint(status)
    parent = ROOT / spec["parent"]
    design = read_json(out / "00/design.json")
    spectra, columns = [], []
    for qi, query in enumerate(spec["queries"]):
        model = load_model(ROOT / spec["operator_parent"], query)
        vectors = []
        for path in design:
            if path["role"] != "source" or path["query"] != query:
                continue
            if path["left"]["early_query_token_ids"] != model["query_token_ids"]:
                raise ValueError("query token coordinate mismatch")
            for node in range(3):
                file, v, delta, _ = source_node(parent, path, node, "source")
                if list(delta.shape) != model["p_shape"]:
                    raise ValueError("query physical shape mismatch")
                status["source_nodes_loaded"] += 1
                for arm, value in zip(spec["arms"], v, strict=True):
                    vectors.append(value.ravel())
                    columns.append(
                        {
                            "query": query,
                            "column": len(vectors) - 1,
                            "role": "source",
                            "path": path["path"],
                            "node": node,
                            "arm": arm,
                            "file": str(file.relative_to(ROOT)),
                            "sha256": digest(file),
                            "raw_norm": float(np.linalg.norm(value)),
                        }
                    )
        vraw = np.stack(vectors, axis=1)
        if vraw.shape[1] != 96 or model["D"].shape[1] != 8:
            raise ValueError("fixed source column count required")
        save(
            out / "01/models" / f"{query}_raw.json",
            {
                "query": query,
                "roles": ["source"] * 104,
                "p_shape": model["p_shape"],
                "token_ids": model["query_token_ids"],
            },
            {"D": model["D"], "V": vraw},
        )
        for si, space in enumerate(spec["spaces"]):
            raw = {"D": model["D"], "V": vraw}.get(space)
            if raw is None:
                raw = np.column_stack((model["D"], vraw))
            fitted = T.source_basis(
                raw, ["source"] * raw.shape[1], spec["rank_multiplier"]
            )
            orth_error = (
                float(
                    np.max(
                        abs(
                            fitted["basis"].T @ fitted["basis"] - np.eye(fitted["rank"])
                        )
                    )
                )
                if fitted["rank"]
                else 0.0
            )
            if orth_error > spec["absolute_tolerance"]:
                raise ValueError("thin source basis is not orthonormal")
            meta = {
                "query": query,
                "space": space,
                "rank": fitted["rank"],
                "dimension": raw.shape[0],
                "columns": raw.shape[1],
                "rank_tolerance": fitted["tolerance"],
                "orthogonality_error": orth_error,
                "zero_columns": np.flatnonzero(~fitted["nonzero"]).tolist(),
                "source_frozen_at": now(),
                "random": False,
            }
            save(
                out / "01/models" / f"{query}_{space}.json",
                meta,
                {k: fitted[k] for k in ("basis", "scales", "nonzero", "eigenvalues")},
            )
            status["learned_bases"] += 1
            spectra.extend(
                {
                    "query": query,
                    "space": space,
                    "index": i,
                    "eigenvalue": float(e),
                    "threshold": fitted["tolerance"],
                    "retained": bool(e > fitted["tolerance"]),
                }
                for i, e in enumerate(fitted["eigenvalues"])
            )
            budget.checkpoint(status)
            if space == "D":
                continue
            for seed in spec["random_seeds"]:
                stream_seed = seed + qi * 10000 + si * 1000
                control = T.random_basis(raw.shape[0], fitted["rank"], stream_seed)
                save(
                    out / "01/models" / f"{query}_{space}_random_{seed}.json",
                    {
                        **meta,
                        "space": f"{space}_random_{seed}",
                        "random": True,
                        "declared_seed": seed,
                        "stream_seed": stream_seed,
                    },
                    {"basis": control},
                )
                status["random_bases"] += 1
                budget.checkpoint(status)
        print(
            {
                "query": query,
                "source_nodes": status["source_nodes_loaded"],
                "seconds": status["analysis_seconds"],
            },
            flush=True,
        )
    if (
        status["learned_bases"],
        status["random_bases"],
        status["source_nodes_loaded"],
    ) != (9, 18, 144):
        raise ValueError("source battery incomplete")
    write_csv(out / "01/source_columns.csv", columns)
    write_csv(out / "01/source_spectrum.csv", spectra)
    files = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "01").rglob("*")
        if p.is_file()
    }
    write_json(
        out / "01/source_barrier.json",
        {
            "frozen_at": now(),
            "files": files,
            "source_nodes": 144,
            "application_nodes_loaded": status["application_nodes_loaded"],
            "learned_bases": 9,
            "random_bases": 18,
        },
    )
    status["status"] = "SOURCE_CHECKPOINTED"
    budget.checkpoint(status)
    print(
        {"status": status["status"], "seconds": status["analysis_seconds"]}, flush=True
    )


def evaluate(out):
    """Evaluate every original path and all source spaces after the source barrier."""
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    if status["status"] != "SOURCE_CHECKPOINTED":
        raise ValueError("complete source barrier must precede numerical application")
    barrier_file = out / "01/source_barrier.json"
    barrier = read_json(barrier_file)
    for name, sha in barrier["files"].items():
        if digest(out / name) != sha:
            raise ValueError("source checkpoint changed")
    budget = Budget(out, spec)
    status.update(
        status="APPLICATION_RUNNING",
        application_started_at=now(),
        source_sha256=digest(barrier_file),
    )
    budget.checkpoint(status)
    nodes, integrals = [], []
    replay = 0.0
    parent, design = ROOT / spec["parent"], read_json(out / "00/design.json")
    for query in spec["queries"]:
        models = {}
        for file in sorted((out / "01/models").glob(query + "_*.json")):
            if file.stem.endswith("_raw"):
                continue
            meta, bank = load(file)
            models[meta["space"]] = bank["basis"]
        if len(models) != 9:
            raise ValueError("all nine spaces per query required")
        for path in design:
            if path["query"] != query:
                continue
            group = {key: [] for key in models}
            for node in range(3):
                _, v, delta, previous = source_node(parent, path, node, path["role"])
                if path["role"] == "application":
                    status["application_nodes_loaded"] += 1
                projected = models["D"] @ (models["D"].T @ delta.ravel())
                replay = max(replay, float(np.max(abs(projected - previous.ravel()))))
                for space, basis in models.items():
                    parts = T.action_parts(basis, v.reshape(2, -1), delta)
                    a = np.stack(
                        [parts[k] for k in ("full", "supported", "missing")], axis=1
                    )
                    group[space].append(a)
                    for i, arm in enumerate(spec["arms"]):
                        nodes.append(
                            {
                                "path": path["path"],
                                "role": path["role"],
                                "query": query,
                                "arm": arm,
                                "space": space,
                                "node": node,
                                "t": node / 2,
                                **{
                                    k: float(parts[k][i])
                                    for k in ("full", "supported", "missing")
                                },
                                "displacement_supported_fraction": parts[
                                    "displacement_supported_energy"
                                ]
                                / parts["displacement_energy"],
                                "covector_supported_fraction": float(
                                    parts["covector_supported_energy"][i]
                                    / parts["covector_energy"][i]
                                ),
                            }
                        )
                budget.checkpoint(status)
            for space, values in group.items():
                for rule, a in M.quadrature(np.stack(values)).items():
                    for i, arm in enumerate(spec["arms"]):
                        integrals.append(
                            {
                                "path": path["path"],
                                "role": path["role"],
                                "query": query,
                                "arm": arm,
                                "space": space,
                                "rule": rule,
                                **dict(
                                    zip(
                                        ("full", "supported", "missing"),
                                        map(float, a[i]),
                                        strict=True,
                                    )
                                ),
                            }
                        )
        print(
            {
                "query": query,
                "evaluated_nodes": len(nodes),
                "seconds": status["analysis_seconds"],
            },
            flush=True,
        )
    if replay > spec["absolute_tolerance"] or status["application_nodes_loaded"] != 144:
        raise ValueError(f"complete application/D replay failed: {replay}")
    metrics = []
    for role in ("source", "application"):
        for query in spec["queries"]:
            for arm in spec["arms"]:
                spaces = sorted({r["space"] for r in integrals if r["query"] == query})
                for space in spaces:
                    for rule in ("simpson", "trapezoid"):
                        chosen = [
                            r
                            for r in integrals
                            if (r["role"], r["query"], r["arm"], r["space"], r["rule"])
                            == (role, query, arm, space, rule)
                        ]
                        metrics.append(
                            {
                                "role": role,
                                "query": query,
                                "arm": arm,
                                "space": space,
                                "rule": rule,
                                **T.signed_metrics(
                                    [
                                        [r[k] for k in ("full", "supported", "missing")]
                                        for r in chosen
                                    ]
                                ),
                            }
                        )
    write_csv(out / "02/nodes.csv", nodes)
    write_csv(out / "02/integrals.csv", integrals)
    write_csv(out / "02/metrics.csv", metrics)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM36_EVALUATED_NOT_GLOBAL",
            "node_arm_space_rows": len(nodes),
            "integral_rows": len(integrals),
            "metric_groups": len(metrics),
            "D_projection_max_replay_error": replay,
            "source_sha256": digest(barrier_file),
            "new_model_calls": 0,
            "scope": spec["scope"],
        },
    )
    status["status"] = "EVALUATED"
    budget.checkpoint(status)
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for sub in ("01", "02")
            for p in (out / sub).rglob("*")
            if p.is_file()
        },
    )
    budget.checkpoint(status)
    print(
        {
            "status": "EVALUATED",
            "seconds": status["analysis_seconds"],
            "D_replay": replay,
        },
        flush=True,
    )


def main():
    """Run one explicit finite packet; preserve partial failures and all prior runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "source", "evaluate", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        try:
            if args.stage in ("prepare", "all"):
                prepare(args.out)
            if args.stage in ("source", "all"):
                fit_source(args.out)
            if args.stage in ("evaluate", "all"):
                evaluate(args.out)
        except Exception as error:
            if args.out.exists():
                write_json(
                    args.out / "failure.json",
                    {
                        "at": now(),
                        "type": type(error).__name__,
                        "error": str(error),
                        "new_model_calls": 0,
                    },
                )
            raise


if __name__ == "__main__":
    main()

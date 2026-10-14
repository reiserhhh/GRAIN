#!/usr/bin/env python3
"""HM40 source-only compensation family, then fixed cold midpoint evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_cancellation_transport as C
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_target_active_support import load, save
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_cancellation_transport_v1.json"
PLAN = ROOT / "docs/GRAIN_CANCELLATION_TRANSPORT_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/cancellation_transport_r1"
CODE = (
    "grain_research/construction/math_cancellation_transport.py",
    "scripts/run_grain_math_cancellation_transport.py",
    "scripts/verify_grain_math_cancellation_transport.py",
    "tests/test_grain_math_cancellation_transport.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)
KEYS = ("link", "decoder", "roundoff", "parent_total", "delta_p")


class Budget:
    """Preserve actual wall time across stages without resetting on failure."""

    def __init__(self, out, started=None):
        self.out = out
        self.status = read_json(out / "run_status.json")
        self.previous = self.status["analysis_seconds"]
        self.started = time.monotonic() if started is None else started

    def checkpoint(self):
        """Charge before enforcing the original300-second CPU cap."""
        self.status["analysis_seconds"] = (
            self.previous + time.monotonic() - self.started
        )
        write_json(self.out / "run_status.json", self.status)
        if self.status["analysis_seconds"] > 300:
            raise TimeoutError("fixed300-second combined CPU budget reached")


def prepare(out):
    """Freeze ancestry and design before opening numerical source/application arrays."""
    started = time.monotonic()
    if out.exists():
        raise FileExistsError("preserve every prior packet")
    spec = read_json(SPEC)
    C.assert_scope(spec)
    parent = ROOT / spec["parent"]
    m = check(parent)
    vf = ROOT / spec["verification"]
    if read_json(vf)["status"] != "VERIFIED_FINITE_HM39_NOT_GLOBAL":
        raise ValueError("verified HM39 required")
    files = dict(m["files"])
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("changed HM39 artifact " + name)
        files[str((parent / name).relative_to(ROOT))] = sha
    companion = ROOT / spec["companion_manifest"]
    for name, sha in read_json(companion)["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("changed HM39 numerical companion " + name)
        files[name] = sha
    for p in (vf, companion):
        files[str(p.relative_to(ROOT))] = digest(p)
    for name in CODE:
        files[name] = digest(ROOT / name)
    design = read_json(parent / "00/design.json")
    if len(design) != 96 or sum(x["role"] == "source" for x in design) != 48:
        raise ValueError("original96paths and48source nodes required")
    for folder in ("00/code", "01/models", "02"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name in CODE:
        p = out / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    write_json(out / "00/design.json", design)
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "frozen_at": now(),
            "new_model_calls": 0,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "analysis_seconds": 0.0,
            "learned_bases": 0,
            "random_bases": 0,
            "source_nodes_loaded": 0,
            "source_nodes_evaluated": 0,
            "application_nodes_loaded": 0,
            "new_model_calls": 0,
        },
    )
    Budget(out, started).checkpoint()
    print({"status": "PREPARED", "paths": 96}, flush=True)


def arrays(parent, path, role, source_barrier=False):
    """Require role and exact semantic address before any numerical file is loaded."""
    if path["role"] != role or (role == "application" and not source_barrier):
        raise ValueError("source-first numerical access required")
    file = parent / "01/nodes" / f"path_{path['path']:03d}_node_1.json"
    meta = array_unit(file)
    if meta["address"] != path or meta["node"] != 1:
        raise ValueError("original midpoint address required")
    with np.load(file.with_suffix(".npz"), allow_pickle=False) as a:
        return {k: a[k].astype(float) for k in KEYS}


def source(out):
    """Construct three source compensation bases and same-rank random controls only."""
    started = time.monotonic()
    m, budget = check(out), Budget(out, started)
    spec, status = m["spec"], budget.status
    if status["status"] != "PREPARED":
        raise ValueError("untouched source stage required, no automatic retry")
    status["status"] = "SOURCE_RUNNING"
    budget.checkpoint()
    parent, design = ROOT / spec["parent"], read_json(out / "00/design.json")
    spectra = []
    for qi, query in enumerate(spec["queries"]):
        _, old = load(ROOT / spec["basis_parent"] / "01/models" / f"{query}_V.json")
        u = old["basis"]
        cs, proxies, addresses = [], [], []
        for path in design:
            if path["query"] != query or path["role"] != "source":
                continue
            a = arrays(parent, path, "source")
            cr = C.compensation(
                a["link"].reshape(2, -1).T, a["decoder"].reshape(2, -1).T, u
            )
            proxy = C.complement(a["roundoff"].reshape(2, -1).T, u)
            cs.extend(cr["c"].T)
            proxies.extend(proxy.T)
            addresses.extend(
                {"path": path["path"], "arm": arm, "role": "source"}
                for arm in spec["arms"]
            )
            status["source_nodes_loaded"] += 1
            budget.checkpoint()
        columns, proxy = np.stack(cs, axis=1), np.stack(proxies, axis=1)
        if columns.shape[1] != 32:
            raise ValueError("exact32source component columns per query required")
        fitted = C.compensation_basis(
            columns, ["source"] * 32, u, proxy, spec["rank_multiplier"]
        )
        w = fitted["basis"]
        random = C.random_complement(u, fitted["rank"], spec["random_seed"] + qi)
        metadata = {
            "query": query,
            "role": "source",
            "rank": fitted["rank"],
            "old_rank": u.shape[1],
            "dimension": len(u),
            "column_addresses": addresses,
            "rank_tolerance": fitted["tolerance"],
            "rank_multiplier": spec["rank_multiplier"],
            "random_seed": spec["random_seed"] + qi,
            "normalized_roundoff_proxy_floor": fitted[
                "normalized_roundoff_proxy_floor"
            ],
            "physical_roundoff_proxy_floor": fitted["physical_roundoff_proxy_floor"],
            "resolved_against_proxy": int(
                fitted["resolved_against_proxy"][: fitted["rank"]].sum()
            ),
            "proxy_is_certified_component_error_bound": False,
            "orthogonality_error": float(np.max(abs(u.T @ w))) if w.shape[1] else 0.0,
            "application_nodes_loaded": 0,
        }
        save(
            out / "01/models" / f"{query}.json",
            metadata,
            {
                "basis": w,
                "random_basis": random,
                "raw": fitted["raw"],
                "roundoff_proxy": proxy,
                "scales": fitted["scales"],
                "singular_values": fitted["singular_values"],
                "raw_singular_values": fitted["raw_singular_values"],
                "gram_eigenvalues": fitted["eigenvalues"],
                "resolved_against_proxy": fitted["resolved_against_proxy"],
            },
        )
        for j, singular in enumerate(fitted["singular_values"]):
            spectra.append(
                {
                    "query": query,
                    "index": j,
                    "normalized_singular": float(singular),
                    "raw_singular": float(fitted["raw_singular_values"][j]),
                    "machine_rank_retained": j < fitted["rank"],
                    "resolved_against_proxy": bool(fitted["resolved_against_proxy"][j]),
                    "resolved_against_physical_tolerance": bool(
                        fitted["raw_singular_values"][j] > spec["absolute_tolerance"]
                    ),
                    "normalized_proxy_floor": metadata[
                        "normalized_roundoff_proxy_floor"
                    ],
                    "physical_proxy_floor": metadata["physical_roundoff_proxy_floor"],
                }
            )
        status["learned_bases"] += 1
        status["random_bases"] += 1
        budget.checkpoint()
        if status["learned_bases"] > 3 or status["random_bases"] > 3:
            raise ValueError("source basis cap exceeded")
        print(
            {
                "query": query,
                "rank": fitted["rank"],
                "resolved_proxy": metadata["resolved_against_proxy"],
            },
            flush=True,
        )
    C.assert_scope(spec, source_nodes=status["source_nodes_loaded"])
    write_csv(out / "01/spectra.csv", spectra)
    files = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "01").rglob("*")
        if p.is_file() and not p.name.startswith("._")
    }
    write_json(
        out / "01/source_barrier.json",
        {
            "files": files,
            "frozen_at": now(),
            "source_nodes": 48,
            "application_nodes_loaded": 0,
            "new_model_calls": 0,
        },
    )
    status["status"] = "SOURCE_SEALED"
    budget.checkpoint()
    print(
        {"status": "SOURCE_SEALED", "analysis_seconds": status["analysis_seconds"]},
        flush=True,
    )


def paired(rows, wordings):
    """Keep every condition and two-wording mean/contrast in their original units."""
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("query", "role", "cell", "arm"))
        groups.setdefault(key, {})[row["wording"]] = row
    result = []
    for key, group in groups.items():
        a, b = (group[w] for w in wordings)
        for measure, weights in (
            ("pair_mean", (0.5, 0.5)),
            ("wording_contrast", (0.5, -0.5)),
        ):
            head = dict(zip(("query", "role", "cell", "arm"), key, strict=True))
            result.append(
                {
                    **head,
                    "measure": measure,
                    **{
                        k: weights[0] * a[k] + weights[1] * b[k]
                        for k in C.ACTION_FIELDS
                    },
                }
            )
    return result


def coverage(rows, head, floor):
    """Separate average covector/input energy coverage from signed action error."""
    result = dict(head, count=len(rows))
    for kind in ("covector", "input"):
        denominator = float(np.mean([r[kind + "_energy"] for r in rows]))
        resolved = denominator > floor**2
        result[kind + "_energy"] = denominator
        result[kind + "_denominator_resolved"] = resolved
        for space in ("u", "w", "random"):
            energy = float(np.mean([r[space + "_" + kind + "_energy"] for r in rows]))
            result[space + "_" + kind + "_energy"] = energy
            result[space + "_" + kind + "_fraction"] = (
                energy / denominator if resolved else None
            )
    return result


def evaluate(out):
    """Evaluate cold actual derivatives only after all source spaces are disk-sealed."""
    started = time.monotonic()
    m, budget = check(out), Budget(out, started)
    spec, status = m["spec"], budget.status
    barrier = out / "01/source_barrier.json"
    C.assert_scope(
        spec,
        source_nodes=status["source_nodes_loaded"],
        application=True,
        barrier=barrier.is_file(),
    )
    if status["status"] != "SOURCE_SEALED":
        raise ValueError("untouched evaluation stage required")
    for name, sha in read_json(barrier)["files"].items():
        if digest(out / name) != sha:
            raise ValueError("source space changed before application")
    status["status"] = "EVALUATING"
    budget.checkpoint()
    rows = []
    parent, design = ROOT / spec["parent"], read_json(out / "00/design.json")
    for query in spec["queries"]:
        _, old = load(ROOT / spec["basis_parent"] / "01/models" / f"{query}_V.json")
        _, bank = load(out / "01/models" / f"{query}.json")
        for role in ("source", "application"):
            for path in design:
                if path["query"] != query or path["role"] != role:
                    continue
                a = arrays(parent, path, role, source_barrier=True)
                parts = C.parts(
                    *(
                        a[k].reshape(2, -1)
                        for k in ("parent_total", "link", "decoder", "roundoff")
                    ),
                    a["delta_p"],
                    old["basis"],
                    bank["basis"],
                    bank["random_basis"],
                )
                for ai, arm in enumerate(spec["arms"]):
                    head = {
                        k: path[k]
                        for k in ("query", "role", "wording", "path", "H", "B", "Q")
                    }
                    head.update(cell=path["H"] * 4 + path["B"] * 2 + path["Q"], arm=arm)
                    rows.append(
                        {
                            **head,
                            **{
                                k: float(v[ai]) if v[ai] is not None else None
                                for k, v in parts.items()
                            },
                        }
                    )
                status[
                    "source_nodes_evaluated"
                    if role == "source"
                    else "application_nodes_loaded"
                ] += 1
                budget.checkpoint()
    pairs = paired(rows, spec["wordings"])
    metric, cover = [], []
    for query in spec["queries"]:
        for role in ("source", "application"):
            for arm in spec["arms"]:
                head = {"query": query, "role": role, "arm": arm}
                select = lambda data, head=head: [
                    r for r in data if all(r[k] == v for k, v in head.items())
                ]
                selected = select(rows)
                cover.append(coverage(selected, head, spec["absolute_tolerance"]))
                metric.append(
                    {**head, "measure": "wording", **C.signed_summary(selected)}
                )
                for measure in ("pair_mean", "wording_contrast"):
                    metric.append(
                        {
                            **head,
                            "measure": measure,
                            **C.signed_summary(
                                [r for r in select(pairs) if r["measure"] == measure]
                            ),
                        }
                    )
    write_csv(out / "02/observations.csv", rows)
    write_csv(out / "02/paired_actions.csv", pairs)
    write_csv(out / "02/action_metrics.csv", metric)
    write_csv(out / "02/coverage.csv", cover)
    maximum = max(r["component_identity_max"] for r in rows)
    if maximum > spec["absolute_tolerance"] or len(rows) != 192 or len(pairs) != 192:
        raise ValueError("full physical component identity and scope required")
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM40_EVALUATED_NOT_GLOBAL",
            "nodes": 96,
            "observations": len(rows),
            "paired_actions": len(pairs),
            "metric_groups": len(metric),
            "coverage_groups": len(cover),
            "source_barrier_sha256": digest(barrier),
            "maximum_component_identity_error": maximum,
            "learned_bases": 3,
            "random_bases": 3,
            "new_model_calls": 0,
            "scope": spec["scope"],
        },
    )
    hashes = {
        str(p.relative_to(out)): digest(p)
        for folder in (out / "01", out / "02")
        for p in folder.rglob("*")
        if p.is_file()
        and p.name != "artifact_hashes.json"
        and not p.name.startswith("._")
    }
    write_json(out / "02/artifact_hashes.json", hashes)
    status["status"] = "EVALUATED"
    budget.checkpoint()
    print(
        {
            "status": "EVALUATED",
            "analysis_seconds": status["analysis_seconds"],
            "observations": len(rows),
        },
        flush=True,
    )


def main():
    """Only explicit bounded stages, preserving failures instead of automatic retries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "source", "evaluate"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=2), collector_guard(args.out):
        started = time.monotonic()
        existed = (args.out / "run_status.json").exists()
        previous = (
            read_json(args.out / "run_status.json")["analysis_seconds"]
            if existed
            else 0.0
        )
        try:
            {"prepare": prepare, "source": source, "evaluate": evaluate}[args.stage](
                args.out
            )
        except Exception as exc:
            if (args.out / "run_status.json").exists():
                state = read_json(args.out / "run_status.json")
                if state["status"] in ("SOURCE_RUNNING", "EVALUATING") or not existed:
                    state.update(
                        status="STOPPED_FAILURE_OR_BUDGET",
                        error=repr(exc),
                        analysis_seconds=max(
                            state["analysis_seconds"],
                            previous + time.monotonic() - started,
                        ),
                    )
                    write_json(args.out / "run_status.json", state)
            raise


if __name__ == "__main__":
    main()

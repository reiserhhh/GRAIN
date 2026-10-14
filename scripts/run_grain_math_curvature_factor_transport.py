#!/usr/bin/env python3
"""HM41 source-first factor bank and complete finite analytic substitution battery."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.monotonic()

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_curvature_factor_transport as F
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_target_active_support import load, save
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_curvature_factor_transport_v1.json"
PLAN = ROOT / "docs/GRAIN_CURVATURE_FACTOR_TRANSPORT_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/curvature_factor_transport_r1"
CODE = (
    "grain_research/construction/math_curvature_factor_transport.py",
    "scripts/run_grain_math_curvature_factor_transport.py",
    "scripts/verify_grain_math_curvature_factor_transport.py",
    "tests/test_grain_math_curvature_factor_transport.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)
FACTOR_KEYS = ("hessian", "h_jacobian", "p_jacobian")
KEYS = (
    *FACTOR_KEYS,
    "logits",
    "probabilities",
    "output_gradient",
    "link",
    "decoder",
    "roundoff",
    "parent_total",
    "delta_h",
    "delta_p",
)


class Budget:
    """Charge actual process time, including failed stages, against one fixed cap."""

    def __init__(self, out, started=PROCESS_STARTED):
        self.out, self.started = out, started
        self.status = read_json(out / "run_status.json")
        self.previous = self.status["analysis_seconds"]

    def checkpoint(self):
        """Save accumulated time before refusing an exhausted budget."""
        self.status["analysis_seconds"] = (
            self.previous + time.monotonic() - self.started
        )
        write_json(self.out / "run_status.json", self.status)
        if self.status["analysis_seconds"] > 300:
            raise TimeoutError("fixed combined300-second CPU budget reached")


def cell(path):
    """Bind a query and all three condition bits to one source reference."""
    return f"{path['query']}_cell_{path['H']}{path['B']}{path['Q']}"


def layout(path):
    """Record absolute positions while aligning only declared relative ordered slots."""
    result = {}
    for side in ("left", "right"):
        result[side] = {
            "h_positions": path[side]["spans"]["A_property"]["positions"],
            "p_positions": path[side]["early_query_positions"],
        }
    if len(result["left"]["h_positions"]) != 4 or result["left"] != result["right"]:
        raise ValueError("original within-path H/P coordinates must align")
    return result


def read_node(parent, path, role, barrier=False, keys=KEYS):
    """Require frozen role/address and a source barrier before numerical application."""
    if path["role"] != role or (role == "application" and not barrier):
        raise ValueError("source-first numerical access required")
    file = parent / "01/nodes" / f"path_{path['path']:03d}_node_1.json"
    meta = array_unit(file)
    if meta["address"] != path or meta["node"] != 1 or meta["t"] != 0.5:
        raise ValueError("original midpoint address required")
    loc = layout(path)["left"]
    with np.load(file.with_suffix(".npz"), allow_pickle=False) as a:
        result = {k: a[k].copy() for k in keys}
    if result["h_jacobian"].shape != (10, 4, 1536) or result["p_jacobian"].shape != (
        10,
        len(loc["p_positions"]),
        1536,
    ):
        raise ValueError("query-specific physical slots differ")
    return result


def prepare(out):
    """Seal inherited scientific objects and code before numerical source access."""
    if out.exists():
        raise FileExistsError("preserve every prior packet")
    spec = read_json(SPEC)
    F.assert_scope(spec)
    previous = ROOT / spec["previous_packet"]
    inherited = check(previous)
    vf = ROOT / spec["previous_verification"]
    if read_json(vf)["status"] != "VERIFIED_FINITE_HM40_NOT_GLOBAL":
        raise ValueError("verified HM40 required")
    files = dict(inherited["files"])
    for base, ledger in (
        (previous, previous / "02/artifact_hashes.json"),
        (
            ROOT / spec["sensitivity"],
            ROOT / spec["sensitivity"] / "artifact_hashes.json",
        ),
    ):
        for rel, sha in read_json(ledger).items():
            if digest(base / rel) != sha:
                raise ValueError("changed prior result " + rel)
            files[str((base / rel).relative_to(ROOT))] = sha
        files[str(ledger.relative_to(ROOT))] = digest(ledger)
    sensitivity = ROOT / spec["sensitivity"] / "source_barrier.json"
    for rel, sha in read_json(sensitivity)["files"].items():
        if digest(ROOT / rel) != sha:
            raise ValueError("changed source sensitivity " + rel)
        files[rel] = sha
    files[str(vf.relative_to(ROOT))] = digest(vf)
    for rel in CODE:
        files[rel] = digest(ROOT / rel)
    design = read_json(ROOT / spec["parent"] / "00/design.json")
    if len(design) != 96 or sum(p["role"] == "source" for p in design) != 48:
        raise ValueError("original96midpoints required")
    for p in design:
        layout(p)
    for folder in ("00/code", "01/models", "02/nodes"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for rel in CODE:
        dest = out / "00/code" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
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
            "source_nodes_loaded": 0,
            "source_bundles": 0,
            "source_nodes_evaluated": 0,
            "application_nodes_loaded": 0,
            "new_model_calls": 0,
            "new_fits": 0,
        },
    )
    Budget(out).checkpoint()
    print({"status": "PREPARED", "nodes": 96}, flush=True)


def source(out):
    """Freeze24two-wording source references with original joint factor triples."""
    m, budget = check(out), Budget(out)
    if budget.status["status"] != "PREPARED":
        raise ValueError("untouched source stage required")
    budget.status["status"] = "SOURCE_RUNNING"
    budget.checkpoint()
    design = read_json(out / "00/design.json")
    parent = ROOT / m["spec"]["parent"]
    for name in sorted({cell(p) for p in design if p["role"] == "source"}):
        paths = sorted(
            [p for p in design if p["role"] == "source" and cell(p) == name],
            key=lambda p: p["wording"],
        )
        if [p["wording"] for p in paths] != [0, 2]:
            raise ValueError("both original source wordings required")
        nodes = [read_node(parent, p, "source", keys=FACTOR_KEYS) for p in paths]
        reference = F.source_reference(
            [tuple(a[k] for k in FACTOR_KEYS) for a in nodes],
            [p["role"] for p in paths],
        )
        for k in ("source_h", "source_p"):
            reference[k] = reference[k].astype(np.float32)
        save(
            out / "01/models" / f"{name}.json",
            {
                "cell": name,
                "query": paths[0]["query"],
                "role": "source",
                "paths": [p["path"] for p in paths],
                "layouts": [layout(p) for p in paths],
                "relative_slot_alignment": True,
                "application_nodes_loaded": 0,
            },
            reference,
        )
        budget.status["source_nodes_loaded"] += 2
        budget.status["source_bundles"] += 1
        budget.checkpoint()
    if (
        budget.status["source_bundles"] != 24
        or budget.status["source_nodes_loaded"] != 48
    ):
        raise ValueError("complete source bank required")
    write_json(
        out / "01/source_barrier.json",
        {
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in (out / "01/models").iterdir()
                if p.is_file() and not p.name.startswith("._")
            },
            "frozen_at": now(),
            "source_nodes": 48,
            "source_bundles": 24,
            "application_nodes_loaded": 0,
            "new_model_calls": 0,
        },
    )
    budget.status["status"] = "SOURCE_SEALED"
    budget.checkpoint()
    print({"status": "SOURCE_SEALED", "bundles": 24}, flush=True)


def node_result(reference, a):
    """Compose every local/source factor, retaining all decoder and pairing terms."""
    actual = tuple(a[k] for k in FACTOR_KEYS)
    cube = F.crossed(reference, actual, a["delta_h"])
    terms = F.mobius(cube)
    direct = F.direct_terms(reference, actual, a["delta_h"])
    paired = F.paired_source(reference, a["delta_h"])
    expected = a["link"].astype(float).reshape(2, -1)
    errors = {
        "link_replay": float(np.max(abs(cube[7] - expected))),
        "direct_terms": float(np.max(abs(terms - direct))),
        "cube_sum": float(np.max(abs(terms.sum(axis=0) - cube[7]))),
    }
    if max(errors.values()) > 1e-8:
        raise ValueError("original physical1e-8replay required")
    dp = a["delta_p"].ravel()
    actions, ma, da = cube @ dp, terms @ dp, direct @ dp
    full, decoder, roundoff = [
        a[k].astype(float).reshape(2, -1) @ dp
        for k in ("parent_total", "decoder", "roundoff")
    ]
    ps = paired @ dp
    rows = []
    for i in range(2):
        row = {
            "full": float(full[i]),
            "link": float(actions[7, i]),
            "decoder": float(decoder[i]),
            "roundoff": float(roundoff[i]),
            "source_mean": float(actions[0, i]),
            "source_paired": float(ps[i]),
            "pairing_loss": float(ps[i] - actions[0, i]),
            "delta_link": float(actions[7, i] - actions[0, i]),
            "error_mean": float(actions[7, i] - actions[0, i]),
            "error_paired": float(actions[7, i] - ps[i]),
            "full_error_mean": float(full[i] - actions[0, i]),
            "full_error_paired": float(full[i] - ps[i]),
        }
        row.update({f"m{j}": float(ma[j, i]) for j in range(8)})
        row.update(
            {
                "error_" + k: float(actions[7, i] - actions[j, i])
                for k, j in (
                    ("C", 1),
                    ("H", 2),
                    ("P", 4),
                    ("CH", 3),
                    ("CP", 5),
                    ("HP", 6),
                )
            }
        )
        rows.append(row)
    return cube, paired, actions, ma, da, rows, errors


def descriptors(a):
    """Report actual reader state, signed C spectrum and physical factor norms."""
    p, c, g = (
        a[k].astype(float) for k in ("probabilities", "hessian", "output_gradient")
    )
    ev = np.linalg.eigvalsh((c + c.T) / 2)
    result = {
        "entropy": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
        "top_probability": float(p.max()),
        "probability_sum": float(p.sum()),
        "g_norm": float(np.linalg.norm(g)),
        "g_gauge_sum": float(g.sum()),
        "C_norm": float(np.linalg.norm(c)),
        "C_gauge_norm": float(np.linalg.norm(c @ np.ones(10))),
        "C_asymmetry": float(np.max(abs(c - c.T))),
        "J_H_norm": float(np.linalg.norm(a["h_jacobian"].astype(float))),
        "J_P_norm": float(np.linalg.norm(a["p_jacobian"].astype(float))),
    }
    result.update({f"C_eigen_{i}": float(x) for i, x in enumerate(ev)})
    result.update(
        {f"H_norm_{i}": float(np.linalg.norm(x)) for i, x in enumerate(a["delta_h"])}
    )
    return result


def paired_rows(rows):
    """Retain each addressed wording mean and contrast, including source error."""
    result = []
    keys = ("query", "role", "H", "B", "Q", "arm")
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected = sorted(
            [r for r in rows if tuple(r[k] for k in keys) == key],
            key=lambda r: r["wording"],
        )
        if [r["wording"] for r in selected] != [0, 2]:
            raise ValueError("complete addressed wording pair required")
        for measure, sign in (("mean", 1), ("contrast", -1)):
            result.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "measure": measure,
                    **{
                        k: (selected[0][k] + sign * selected[1][k]) / 2
                        for k in F.FIELDS
                    },
                }
            )
    return result


def metrics(rows, pairs):
    """Report all word/mean/contrast groups; do not select a favorable target."""
    result = []
    keys = ("query", "role", "arm")
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        for measure in ("word", "mean", "contrast"):
            candidates = (
                rows
                if measure == "word"
                else [r for r in pairs if r["measure"] == measure]
            )
            group = [r for r in candidates if tuple(r[k] for k in keys) == key]
            result.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "measure": measure,
                    **F.moment_summary(group),
                }
            )
    return result


def evaluate(out):
    """Evaluate the frozen source family and actual-factor diagnostics on all96nodes."""
    m, budget = check(out), Budget(out)
    if budget.status["status"] != "SOURCE_SEALED":
        raise ValueError("source bank must already be sealed")
    barrier = out / "01/source_barrier.json"
    for rel, sha in read_json(barrier)["files"].items():
        if digest(out / rel) != sha:
            raise ValueError("source factor seal changed")
    budget.status["status"] = "EVALUATING"
    budget.checkpoint()
    design = read_json(out / "00/design.json")
    parent, rows, crossings, desc = ROOT / m["spec"]["parent"], [], [], []
    for query in m["spec"]["queries"]:
        cache = {}
        for path in [p for p in design if p["query"] == query]:
            name = cell(path)
            if name not in cache:
                cache[name] = load(out / "01/models" / f"{name}.json")[1]
            a = read_node(parent, path, path["role"], barrier=True)
            cube, paired, ca, ma, da, scalar, errors = node_result(cache[name], a)
            head = {
                k: path[k] for k in ("path", "query", "role", "wording", "H", "B", "Q")
            }
            save(
                out / "02/nodes" / f"path_{path['path']:03d}.json",
                {
                    "address": path,
                    "layouts": layout(path),
                    "source_cell": name,
                    "source_barrier_sha256": digest(barrier),
                    "replay": errors,
                    "source_model_sha256": digest(out / "01/models" / f"{name}.json"),
                },
                {"crossed": cube, "paired_source": paired},
            )
            desc.append({**head, **descriptors(a)})
            for i, arm in enumerate(m["spec"]["arms"]):
                rows.append({**head, "arm": arm, **scalar[i]})
                for mask in range(8):
                    crossings.append(
                        {
                            **head,
                            "arm": arm,
                            "mask": mask,
                            "is_local_factor_diagnostic": bool(mask),
                            "crossed_action": float(ca[mask, i]),
                            "mobius_action": float(ma[mask, i]),
                            "direct_delta_action": float(da[mask, i]),
                        }
                    )
            budget.status[
                "source_nodes_evaluated"
                if path["role"] == "source"
                else "application_nodes_loaded"
            ] += 1
            budget.checkpoint()
        print({"query": query, "nodes": 32}, flush=True)
    pairs = paired_rows(rows)
    moment = metrics(rows, pairs)
    if (len(rows), len(crossings), len(pairs), len(moment), len(desc)) != (
        192,
        1536,
        192,
        36,
        96,
    ):
        raise ValueError("full scientific design required")
    for name, data in (
        ("observations", rows),
        ("crossings", crossings),
        ("paired_actions", pairs),
        ("action_metrics", moment),
        ("factor_descriptors", desc),
    ):
        write_csv(out / "02" / f"{name}.csv", data)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM41_EVALUATED_NOT_GLOBAL",
            "nodes": 96,
            "observations": 192,
            "crossings": 1536,
            "paired_actions": 192,
            "metric_groups": 36,
            "factor_descriptors": 96,
            "source_bundles": 24,
            "source_barrier_sha256": digest(barrier),
            "new_model_calls": 0,
            "new_fits": 0,
            "scope": m["spec"]["scope"],
        },
    )
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for folder in (out / "01", out / "02")
            for p in folder.rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    budget.status["status"] = "EVALUATED"
    budget.checkpoint()
    print(budget.status, flush=True)


def main():
    """Run one explicit stage; never retry a scientific failure automatically."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "source", "evaluate"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    existed = (args.out / "run_status.json").exists()
    previous = (
        read_json(args.out / "run_status.json")["analysis_seconds"] if existed else 0.0
    )
    old = signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("fixed CPU cap"))
    )
    signal.setitimer(
        signal.ITIMER_REAL,
        max(0.001, 300 - previous - (time.monotonic() - PROCESS_STARTED)),
    )
    try:
        with threadpool_limits(limits=2), collector_guard(args.out):
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
                        previous + time.monotonic() - PROCESS_STARTED,
                    ),
                )
                write_json(args.out / "run_status.json", state)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


if __name__ == "__main__":
    main()

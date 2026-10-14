#!/usr/bin/env python3
"""HM39 fixed-midpoint composition: actual output-link versus decoder curvature."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_active_reading_pairing as PAIR
from grain_research.construction import math_mixed_query_path as MIX
from grain_research.construction import math_query_context as CONTEXT
from grain_research.construction import math_readout_curvature as R
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_mixed_query_path as OLD
from scripts import run_grain_math_target_active_support as ARR
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_readout_curvature_v1.json"
PLAN = ROOT / "docs/GRAIN_READOUT_DECODER_CURVATURE_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/readout_curvature_r1"
CODE = (
    "grain_research/construction/math_readout_curvature.py",
    "scripts/run_grain_math_readout_curvature.py",
    "scripts/verify_grain_math_readout_curvature.py",
    "tests/test_grain_math_readout_curvature.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def path_file(out, path):
    """One unique midpoint artifact per inherited semantic address."""
    return out / "01/nodes" / f"path_{path:03d}_node_1.json"


def prepare(out):
    """Hash-seal code, complete ancestry and unchanged96path design before calls."""
    start = time.monotonic()
    if out.exists():
        raise FileExistsError("preserve previous packet")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    pm = check(parent)
    vf = ROOT / spec["verification"]
    if read_json(vf)["status"] != "VERIFIED_FINITE_HM38_NOT_GLOBAL":
        raise ValueError("verified HM38 required")
    files = dict(pm["files"])
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("changed parent artifact " + name)
        files[str((parent / name).relative_to(ROOT))] = sha
    files[str(vf.relative_to(ROOT))] = digest(vf)
    design = read_json(parent / "00/design.json")
    if len(design) != 96 or sum(p["role"] == "source" for p in design) != 48:
        raise ValueError("unaltered finite design required")
    for name in CODE:
        files[name] = digest(ROOT / name)
    for sub in ("00/code", "01/nodes", "01/references", "02"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    mixed = read_json(ROOT / spec["mixed_parent"] / "00/manifest.json")
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "spec": spec,
            "alphabet": mixed["alphabet"],
            "frozen_at": now(),
            "new_fits": 0,
        },
    )
    status = MIX.new_status()
    status["analysis_seconds"] = time.monotonic() - start
    write_json(out / "run_status.json", status)
    if status["analysis_seconds"] > spec["maximum_analysis_seconds"]:
        raise TimeoutError("prepare exhausted CPU budget")
    print({"status": "PREPARED", "paths": len(design)}, flush=True)


def observe(observer, row, m, receiver, donors, delta_h, journal):
    """One forward, ten plain Jacobians, one detached weighted gradient, two mixed gradients."""
    import torch

    hp, pp = row["spans"]["A_property"]["positions"], row["early_query_positions"]
    R.validate_layout(row, donors["h"].shape, donors["p"].shape)
    device, slots = next(observer.model.parameters()).device, {}

    def hook(_module, _args, output):
        leaf, audit = CONTEXT.replace_leaf(
            hidden_tensor(output), hp, pp, receiver, donors, True
        )
        slots.update(leaf=leaf, audit=audit)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    handle = observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(hook)
    try:
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        journal("forward", "attempted")
        output = observer.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        observer.calls += 1
        torch.mps.synchronize()
        journal("forward", "completed")
        full_logits = output.logits[0, -1]
        logits = full_logits[m["alphabet"]["token_ids"]].float()
        values = torch.tensor(
            m["alphabet"]["values"], dtype=torch.float32, device=device
        )
        p = torch.softmax(logits, 0)
        score = (p * values).sum()
        g = (p * (values - score)).detach()
        h_jacobian, p_jacobian = [], []
        for value in logits:
            journal("first", "attempted")
            jac = torch.autograd.grad(value, slots["leaf"], retain_graph=True)[0]
            torch.mps.synchronize()
            journal("first", "completed")
            h_jacobian.append(jac[0, hp].detach().cpu().numpy().copy())
            p_jacobian.append(jac[0, pp].detach().cpu().numpy().copy())
        journal("first", "attempted")
        gradient = torch.autograd.grad(
            (g * logits).sum(), slots["leaf"], create_graph=True, retain_graph=True
        )[0][0, hp]
        torch.mps.synchronize()
        journal("first", "completed")
        decoder = []
        for i, dh in enumerate(delta_h):
            scalar = (
                gradient * torch.tensor(dh, dtype=gradient.dtype, device=device)
            ).sum()
            journal("second", "attempted")
            second = torch.autograd.grad(scalar, slots["leaf"], retain_graph=i == 0)[0][
                0, pp
            ]
            torch.mps.synchronize()
            journal("second", "completed")
            decoder.append(second.detach().cpu().numpy().copy())
        arrays = {
            "logits": logits.detach().cpu().numpy().copy(),
            "probabilities": p.detach().cpu().numpy().copy(),
            "output_gradient": g.cpu().numpy().copy(),
            "h_jacobian": np.stack(h_jacobian),
            "p_jacobian": np.stack(p_jacobian),
            "gradient": gradient.detach().cpu().numpy().copy(),
            "decoder": np.stack(decoder),
        }
        arrays["hessian"] = R.hessian_from_actual(
            arrays["probabilities"], arrays["output_gradient"]
        )
        arrays["h_directional"] = np.einsum(
            "jhd,ahd->aj", arrays["h_jacobian"].astype(float), delta_h
        )
        arrays["link"] = R.link_covectors(
            arrays["hessian"], arrays["h_directional"], arrays["p_jacobian"]
        )
        arrays["total"] = arrays["link"] + arrays["decoder"]
        return (
            arrays,
            native_readout(full_logits.detach().float().cpu().numpy(), m["alphabet"]),
            float(score.detach().cpu()),
            slots["audit"],
        )
    finally:
        handle.remove()


def acquire(out, source):
    """Acquire one finite stage, preserving every attempt and refusing uncertain replay."""
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    expected = "PREPARED" if source else "SOURCE_CHECKPOINTED"
    if status["status"] != expected or status["in_flight"] is not None:
        raise ValueError("untouched explicit stage required; no automatic retry")
    barrier = out / "01/source_barrier.json"
    if barrier.exists() == source:
        raise ValueError("source acquisition and checkpoint must precede application")
    source_sha = None if source else digest(barrier)
    if not source:
        for name, sha in read_json(barrier)["files"].items():
            if digest(out / name) != sha:
                raise ValueError("changed source checkpoint")
    started, previous = time.monotonic(), status["model_seconds"]

    def persist():
        status["model_seconds"] = previous + time.monotonic() - started
        write_json(out / "run_status.json", status)

    def journal(kind, phase):
        nonlocal status
        if phase == "attempted":
            if previous + time.monotonic() - started >= spec["maximum_model_seconds"]:
                raise TimeoutError("fixed cumulative model-time cap reached")
            status = MIX.reserve(status, kind, spec[f"maximum_{kind}_calls"])
        else:
            status[f"{kind}_completed"] += 1
        persist()

    try:
        observer = load_observer()
        if (
            next(observer.model.parameters()).device.type != spec["device"]
            or observer.model.config._attn_implementation != spec["attention_backend"]
            or any(p.requires_grad for p in observer.model.parameters())
        ):
            raise ValueError("unchanged frozen eager MPS reader required")
        write_json(
            out
            / ("01/source_observer.json" if source else "01/application_observer.json"),
            observer.manifest(),
        )
        parent = ROOT / spec["mixed_parent"]
        old_manifest = read_json(parent / "00/manifest.json")
        for path in read_json(out / "00/design.json"):
            if (path["role"] == "source") != source:
                continue
            dst = path_file(out, path["path"])
            if dst.exists() or dst.with_suffix(".npz").exists():
                raise ValueError("existing midpoint cannot be overwritten")
            status.update(
                status="SOURCE_CALLS" if source else "APPLICATION_CALLS",
                in_flight={"path": path["path"], "node": 1, "started_at": now()},
            )
            persist()
            old_meta, old_arrays = ARR.load(path_file(parent, path["path"]))
            if (
                old_meta["address"] != path
                or old_meta["node"] != 1
                or old_meta["t"] != 0.5
            ):
                raise ValueError("identical inherited midpoint required")
            receiver, donors, fixed = OLD.inputs(parent, path, old_manifest)
            donors["p"] = MIX.interpolate(fixed["p0"], fixed["p1"], 0.5)
            for name in ("h", "p", "q", "delta_h"):
                if not np.array_equal(
                    (fixed if name == "delta_h" else donors)[name], old_arrays[name]
                ):
                    raise ValueError("changed physical donor " + name)
            arrays, native, score, audit = observe(
                observer, path["left"], m, receiver, donors, fixed["delta_h"], journal
            )
            replay = {
                "score_error": abs(score - old_meta["gradient_readout"]),
                "gradient_max_error": float(
                    np.max(
                        abs(arrays["gradient"].astype(float) - old_arrays["gradient"])
                    )
                ),
                "covector_max_error": float(
                    np.max(abs(arrays["total"] - old_arrays["p_covectors"]))
                ),
            }
            if (
                native != old_meta["native"]
                or max(replay.values()) > spec["absolute_tolerance"]
                or max(audit.values()) != 0
            ):
                raise ValueError("original1e-8 midpoint replay failed: " + str(replay))
            arrays.update(
                parent_total=old_arrays["p_covectors"],
                delta_p=old_arrays["delta_p"],
                delta_h=fixed["delta_h"],
                norms=np.linalg.norm(fixed["delta_h"].reshape(2, -1), axis=1),
            )
            arrays["roundoff"] = arrays["parent_total"].astype(float) - arrays["total"]
            ARR.save(
                dst,
                {
                    "address": path,
                    "node": 1,
                    "t": 0.5,
                    "native": native,
                    "score": score,
                    "audit": audit,
                    "replay": replay,
                    "parent_file": str(
                        path_file(parent, path["path"]).relative_to(ROOT)
                    ),
                    "parent_sha256": digest(path_file(parent, path["path"])),
                    "source_sha256": source_sha,
                    "acquired_at": now(),
                },
                arrays,
            )
            status["committed_nodes"] += 1
            status["in_flight"] = None
            persist()
            if status["model_seconds"] > spec["maximum_model_seconds"]:
                raise TimeoutError("model budget exceeded during completed-node save")
            if status["committed_nodes"] % 8 == 0:
                print(
                    {
                        "nodes": status["committed_nodes"],
                        "model_seconds": status["model_seconds"],
                        **replay,
                    },
                    flush=True,
                )
        status["status"] = "SOURCE_ACQUIRED" if source else "APPLICATION_ACQUIRED"
        persist()
    except Exception as exc:
        status.update(status="STOPPED_FAILURE_OR_BUDGET", error=repr(exc))
        persist()
        raise


def cpu_save(out, status, start, previous, spec):
    """Charge all checkpoint/analysis wall time to the original finite CPU cap."""
    status["analysis_seconds"] = previous + time.monotonic() - start
    write_json(out / "run_status.json", status)
    if status["analysis_seconds"] > spec["maximum_analysis_seconds"]:
        raise TimeoutError("fixed120-second CPU budget reached")


def seal_source(out):
    """Save component source means, not application-fitted prediction coefficients."""
    start = time.monotonic()
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    previous = status["analysis_seconds"]
    if status["status"] != "SOURCE_ACQUIRED" or status["committed_nodes"] != 48:
        raise ValueError("all48source nodes required before application")
    design = read_json(out / "00/design.json")
    for query in spec["queries"]:
        _, bank = ARR.load(
            ROOT / spec["basis_parent"] / "01/models" / f"{query}_V.json"
        )
        u = bank["basis"]
        theta = np.empty((8, 2, 3, 2, u.shape[1]))
        for path in design:
            if path["query"] != query or path["role"] != "source":
                continue
            _, a = ARR.load(path_file(out, path["path"]))
            cell, wi = (
                path["H"] * 4 + path["B"] * 2 + path["Q"],
                spec["wordings"].index(path["wording"]),
            )
            for ci, name in enumerate(R.COMPONENTS):
                theta[cell, wi, ci] = R.project_components(
                    a[name], u, a["delta_p"], a["norms"]
                )["theta"]
        reference = theta.mean(axis=1)
        for cell in range(8):
            _, old = ARR.load(
                ROOT / spec["parent"] / "01/arrays" / f"{query}_source_cell_{cell}.json"
            )
            if (
                np.max(abs(reference[cell].sum(axis=0) - old["reference"][1]))
                > spec["absolute_tolerance"]
            ):
                raise ValueError(
                    "component source means do not replay old full-law reference"
                )
        ARR.save(
            out / "01/references" / f"{query}.json",
            {
                "query": query,
                "role": "source",
                "components": list(R.COMPONENTS),
                "wordings": spec["wordings"],
                "diagnostic_only": True,
            },
            {"theta": theta, "reference": reference},
        )
    files = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "01").rglob("*")
        if p.is_file() and not p.name.startswith("._")
    }
    write_json(
        out / "01/source_barrier.json",
        {
            "files": files,
            "source_nodes": 48,
            "application_nodes_loaded": 0,
            "frozen_at": now(),
        },
    )
    status["status"] = "SOURCE_CHECKPOINTED"
    cpu_save(out, status, start, previous, spec)
    print(
        {"status": status["status"], "analysis_seconds": status["analysis_seconds"]},
        flush=True,
    )


def analyze(out):
    """Project all signed components and replay the HM38 midpoint decomposition."""
    start = time.monotonic()
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    previous = status["analysis_seconds"]
    if status["status"] != "APPLICATION_ACQUIRED" or status["committed_nodes"] != 96:
        raise ValueError("all96actual captures required")
    for kind, count in (("forward", 96), ("first", 1056), ("second", 192)):
        if status[f"{kind}_attempted"] != count or status[f"{kind}_completed"] != count:
            raise ValueError("complete actual call journal required")
    design = read_json(out / "00/design.json")
    actions, pairs, words = [], [], []
    maximum = 0.0
    for query in spec["queries"]:
        _, bank = ARR.load(
            ROOT / spec["basis_parent"] / "01/models" / f"{query}_V.json"
        )
        _, ref = ARR.load(out / "01/references" / f"{query}.json")
        u = bank["basis"]
        for role in ("source", "application"):
            for cell in range(8):
                paths = sorted(
                    [
                        p
                        for p in design
                        if p["query"] == query
                        and p["role"] == role
                        and p["H"] * 4 + p["B"] * 2 + p["Q"] == cell
                    ],
                    key=lambda p: p["wording"],
                )
                _, oldpair = ARR.load(
                    ROOT
                    / spec["parent"]
                    / "01/arrays"
                    / f"{query}_{role}_cell_{cell}.json"
                )
                projections = {name: [] for name in (*R.COMPONENTS, "parent")}
                for path in paths:
                    _, a = ARR.load(path_file(out, path["path"]))
                    for name, items in projections.items():
                        proj = R.project_components(
                            a["parent_total"] if name == "parent" else a[name],
                            u,
                            a["delta_p"],
                            a["norms"],
                        )
                        items.append(proj)
                    for ai, arm in enumerate(spec["arms"]):
                        for observable in ("full", "supported", "complement"):
                            row = {
                                "query": query,
                                "role": role,
                                "cell": cell,
                                "path": path["path"],
                                "wording": path["wording"],
                                "arm": arm,
                                "observable": observable,
                            }
                            row.update(
                                {
                                    name: float(projections[name][-1][observable][ai])
                                    for name in R.COMPONENTS
                                }
                            )
                            row["total"] = float(
                                projections["parent"][-1][observable][ai]
                            )
                            maximum = max(
                                maximum,
                                abs(sum(row[n] for n in R.COMPONENTS) - row["total"]),
                            )
                            actions.append(row)
                for name, projs in projections.items():
                    theta = np.stack([p["theta"] for p in projs], axis=1)
                    eta = np.stack([p["eta"] for p in projs], axis=1)
                    full = np.stack([p["full"] for p in projs], axis=1)
                    reference = (
                        oldpair["reference"][1]
                        if name == "parent"
                        else ref["reference"][cell, R.COMPONENTS.index(name)]
                    )
                    if name == "parent":
                        maximum = max(
                            maximum,
                            float(np.max(abs(theta - oldpair["theta"][1]))),
                            float(np.max(abs(eta - oldpair["eta"][1]))),
                            float(np.max(abs(full - oldpair["full"][1]))),
                        )
                    for ai, arm in enumerate(spec["arms"]):
                        pair, word = PAIR.decompose(
                            theta[ai], eta[ai], full[ai], reference[ai]
                        )
                        head = {
                            "query": query,
                            "role": role,
                            "cell": cell,
                            "arm": arm,
                            "component": name,
                        }
                        pairs.append({**head, **pair})
                        words.extend(
                            {**head, "wording": w, **r}
                            for w, r in zip(spec["wordings"], word, strict=True)
                        )
                cpu_save(out, status, start, previous, spec)
    metric = []
    for query in spec["queries"]:
        for role in ("source", "application"):
            for arm in spec["arms"]:
                head = {"query": query, "role": role, "arm": arm}
                for obs in ("full", "supported", "complement"):
                    selected = [
                        r
                        for r in actions
                        if all(r[k] == v for k, v in head.items())
                        and r["observable"] == obs
                    ]
                    metric.append(
                        {**head, "measure": "action_" + obs, **R.metrics(selected)}
                    )
                for scope, rows, terms in (
                    (
                        "pair",
                        pairs,
                        (
                            "actual",
                            "error",
                            *PAIR.PAIR_COMPONENTS,
                            "coefficient_contrast",
                            "input_contrast",
                        ),
                    ),
                    ("word", words, ("actual", "error", *PAIR.WORD_COMPONENTS)),
                ):
                    selected = [
                        r for r in rows if all(r[k] == v for k, v in head.items())
                    ]
                    for term in terms:
                        group = {}
                        for r in selected:
                            key = (r["cell"], r.get("wording", -1))
                            group.setdefault(key, {})[
                                "total"
                                if r["component"] == "parent"
                                else r["component"]
                            ] = r[term]
                        result = R.metrics(list(group.values()))
                        maximum = max(maximum, result["sum_identity_error"])
                        metric.append({**head, "measure": scope + "_" + term, **result})
    if maximum > spec["absolute_tolerance"]:
        raise ValueError("physical component/HM38 replay failed: " + str(maximum))
    write_csv(out / "02/actions.csv", actions)
    write_csv(out / "02/pair_decomposition.csv", pairs)
    write_csv(out / "02/word_decomposition.csv", words)
    write_csv(out / "02/metrics.csv", metric)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM39_EVALUATED_NOT_GLOBAL",
            "nodes": 96,
            "node_arm_triples": 192,
            "pair_rows": len(pairs),
            "word_rows": len(words),
            "action_rows": len(actions),
            "metric_rows": len(metric),
            "maximum_identity_error": maximum,
            "new_fits": 0,
            "new_neural_training": False,
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
    cpu_save(out, status, start, previous, spec)
    print(
        {
            "status": status["status"],
            "maximum_identity_error": maximum,
            "analysis_seconds": status["analysis_seconds"],
        },
        flush=True,
    )


def main():
    """Explicit bounded stages only; uncertain failures are never auto-resumed."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "source", "seal", "application", "analyze")
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=2), collector_guard(args.out):
        {
            "prepare": prepare,
            "source": lambda o: acquire(o, True),
            "seal": seal_source,
            "application": lambda o: acquire(o, False),
            "analyze": analyze,
        }[args.stage](args.out)


if __name__ == "__main__":
    main()

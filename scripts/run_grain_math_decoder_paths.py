#!/usr/bin/env python3
"""HM25 fixed9-node property path with two readout covectors, no fitting."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_decoder_paths as P
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_decoder_fibers_r2 import check as parent_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_decoder_paths_v1.json"
PLAN = ROOT / "docs/GRAIN_DECODER_PATH_INTEGRAL_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/decoder_paths_r1"
CODE = (
    "grain_research/construction/math_decoder_paths.py",
    "scripts/run_grain_math_decoder_paths.py",
    "scripts/verify_grain_math_decoder_paths.py",
    "tests/test_grain_math_decoder_paths.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out):
    """Check all frozen code, model and input addresses before a resumed stage."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen input changed " + name)
    if digest(out / "00/design.json") != m["design_sha256"]:
        raise ValueError("frozen design changed")
    return m


def prepare(out):
    """Freeze all existing48 references and432 node addresses before any call."""
    if out.exists():
        raise FileExistsError("preserve HM25; resume the existing output")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent_packet"]
    m = parent_check(parent)
    if read_json(ROOT / spec["parent_verification"] / "summary.json")["counts"] != {
        "units": 48,
        "directions": 144,
        "spans": 288,
    }:
        raise ValueError("HM24 complete numerical verification required")
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("HM24 artifact changed " + name)
    design = read_json(parent / "00/design.json")
    P.simpson_weights(spec["path_nodes"])
    if (
        len(design) != spec["units"]
        or len(design) * len(spec["path_nodes"]) != spec["maximum_calls"]
    ):
        raise ValueError("incomplete bounded design")
    files = dict(m["files"])
    files.update(
        {str((parent / name).relative_to(ROOT)): sha for name, sha in ledger.items()}
    )
    files.update({name: digest(ROOT / name) for name in CODE})
    for p in MODEL.iterdir():
        if p.is_file() and p.suffix in (".json", ".safetensors"):
            files[str(p.relative_to(ROOT))] = digest(p)
    (out / "00/code").mkdir(parents=True)
    (out / "01/nodes").mkdir(parents=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        target = out / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "spec": spec,
            "alphabet": m["alphabet"],
            "design_sha256": digest(out / "00/design.json"),
            "all48_HM24_references_no_effect_selection": True,
            "frozen_before_calls": True,
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "nodes": 0, "forward_calls": 0, "backward_calls": 0},
    )
    print(
        {"status": "PREPARED", "units": len(design), "nodes": spec["maximum_calls"]},
        flush=True,
    )


def node_path(out, unit, node):
    """Bind each derivative to its original reference/query and exact path node."""
    return out / "01/nodes" / f"unit_{unit:03d}_node_{node}.json"


def run(out):
    """Run only absent fixed nodes, keeping parameter gradients disabled."""
    import torch

    m = check(out)
    spec, alphabet = m["spec"], m["alphabet"]
    design, ts = read_json(out / "00/design.json"), spec["path_nodes"]
    old = read_json(out / "run_status.json")
    completed = sum(
        node_path(out, row["hm24_unit"], k).exists()
        for row in design
        for k in range(len(ts))
    )
    if completed == spec["maximum_calls"]:
        print({"status": "ALREADY_COMPLETE"}, flush=True)
        return
    if (old["nodes"], old["forward_calls"], old["backward_calls"]) != (
        completed,
        completed,
        2 * completed,
    ):
        raise ValueError(
            "partial-call accounting needs explicit recovery, not blind retry"
        )
    observer = load_observer()
    if any(p.requires_grad for p in observer.model.parameters()):
        raise ValueError("fixed-model parameters must not require gradients")
    write_json(out / "01/observer.json", observer.manifest())
    device = next(observer.model.parameters()).device
    start, backwards = time.monotonic(), 0

    def status(label, observer=observer):
        value = {
            "status": label,
            "nodes": completed,
            "forward_calls": old["forward_calls"] + observer.calls,
            "backward_calls": old["backward_calls"] + backwards,
            "stage_seconds": time.monotonic() - start,
        }
        write_json(out / "run_status.json", value)
        return value

    try:
        for row in design:
            cap, span = row["capture"], row["span"]
            names = tuple(cap["spans"])
            with np.load(ROOT / row["reference_body_npz"]) as cache:
                bodies = {name: cache["L16_" + name].astype(float) for name in names}
            with np.load(ROOT / row["reference_npz"]) as cache:
                question = cache["L16_Q"].astype(float)
            with np.load(ROOT / row["donor_npz"]) as cache:
                donor = cache["L16_" + span].astype(float)
            # Differentiate the continuous ideal path through cached float32 values.
            delta = donor - bodies[span]
            baseline = read_json(ROOT / row["reference_path"])["native"]
            endpoint = read_json(ROOT / row["unit_path"])["native"]
            _, g0 = P.candidate_score_gradient(
                baseline["candidate_logits"], alphabet["values"]
            )
            frozen_covector = torch.tensor(g0, dtype=torch.float32, device=device)
            ids = torch.tensor([cap["input_ids"]], dtype=torch.long, device=device)
            for k, t in enumerate(ts):
                path = node_path(out, row["hm24_unit"], k)
                if path.exists():
                    r = read_json(path)
                    if (
                        r["address"] != row
                        or r["t"] != t
                        or digest(path.with_suffix(".npz")) != r["arrays_sha256"]
                    ):
                        raise ValueError("resumed node changed")
                    continue
                if (
                    old["forward_calls"] + observer.calls >= spec["maximum_calls"]
                    or time.monotonic() - start >= spec["maximum_stage_seconds"]
                ):
                    print(status("RESOURCE_BOUNDARY"), flush=True)
                    return
                patch = (bodies[span] + t * delta).astype(np.float32)
                # Endpoints use the actual cached states, not an interpolated rounding.
                if k == 0:
                    patch = bodies[span].astype(np.float32)
                elif k == len(ts) - 1:
                    patch = donor.astype(np.float32)
                audit, slots = {}, []

                def hook(
                    _module,
                    _args,
                    output,
                    cap=cap,
                    names=names,
                    bodies=bodies,
                    question=question,
                    patch=patch,
                    span=span,
                    audit=audit,
                    slots=slots,
                ):
                    h = hidden_tensor(output)
                    err = max(
                        float(
                            np.max(
                                abs(
                                    h[0, cap["spans"][name]["positions"]]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    - bodies[name]
                                )
                            )
                        )
                        for name in names
                    )
                    err = max(
                        err,
                        float(np.max(abs(h[0, -1].detach().cpu().numpy() - question))),
                    )
                    if err != 0:
                        raise ValueError("live original state changed")
                    audit["live_reference_error"] = err
                    leaf = h.detach().clone()
                    leaf[0, cap["spans"][span]["positions"]] = torch.tensor(
                        patch, dtype=leaf.dtype, device=leaf.device
                    )
                    leaf.requires_grad_(True)
                    slots.append(leaf)
                    return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

                handle = observer.model.model.layers[
                    spec["layer"]
                ].register_forward_hook(hook)
                try:
                    with torch.enable_grad():
                        output = observer.model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            use_cache=False,
                            logits_to_keep=1,
                            return_dict=True,
                        )
                        observer.calls += 1
                        logits = output.logits[0, -1]
                        candidate = logits[
                            torch.tensor(alphabet["token_ids"], device=device)
                        ]
                        score = torch.softmax(candidate, dim=0) @ torch.tensor(
                            alphabet["values"], dtype=candidate.dtype, device=device
                        )
                        linear = candidate @ frozen_covector
                        g_native = (
                            torch.autograd.grad(score, slots[0], retain_graph=True)[0][
                                0
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        backwards += 1
                        g_frozen = (
                            torch.autograd.grad(linear, slots[0])[0][0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        backwards += 1
                        actual = native_readout(logits.detach().cpu().numpy(), alphabet)
                finally:
                    handle.remove()
                expected = (
                    baseline if k == 0 else endpoint if k == len(ts) - 1 else None
                )
                if (
                    expected is not None
                    and expected["candidate_logits"] != actual["candidate_logits"]
                ):
                    raise ValueError(
                        "actual path endpoint differs from frozen native patch"
                    )
                arrays = {
                    "actual_property_state": patch.astype(float),
                    "path_delta": delta,
                }
                for name in names:
                    positions = cap["spans"][name]["positions"]
                    arrays["g_native_" + name] = g_native[positions]
                    arrays["g_frozen_" + name] = g_frozen[positions]
                native_derivative = float(np.sum(arrays["g_native_" + span] * delta))
                frozen_derivative = float(np.sum(arrays["g_frozen_" + span] * delta))
                np.savez_compressed(path.with_suffix(".npz"), **arrays)
                write_json(
                    path,
                    {
                        "address": row,
                        "node": k,
                        "t": t,
                        "native": actual,
                        "audit": audit,
                        "native_derivative": native_derivative,
                        "frozen_derivative": frozen_derivative,
                        "readout_increment_derivative": native_derivative
                        - frozen_derivative,
                        "gradient_native_dtype": "float32 model derivative; float64 contractions/readout",
                        "frozen_readout_covector": g0.tolist(),
                        "arrays_sha256": digest(path.with_suffix(".npz")),
                    },
                )
                del slots, output, logits, candidate, score, linear, g_native, g_frozen
                completed += 1
                status("RUNNING")
            print(status("RUNNING"), flush=True)
        print(status("COMPLETE_FINITE_HM25_NOT_GLOBAL"), flush=True)
    except BaseException as exc:
        write_json(
            out / "failure_run.json", {"error": repr(exc), "state": status("FAILED")}
        )
        raise
    finally:
        del observer
        torch.mps.empty_cache()


def summarize(out):
    """Separate finite decoder and readout increments with fixed quadrature diagnostics."""
    m = check(out)
    ts = np.array(m["spec"]["path_nodes"])
    rows = []
    for address in read_json(out / "00/design.json"):
        nodes = [
            read_json(node_path(out, address["hm24_unit"], k)) for k in range(len(ts))
        ]
        for r in nodes:
            if (
                r["address"] != address
                or digest(
                    node_path(out, address["hm24_unit"], r["node"]).with_suffix(".npz")
                )
                != r["arrays_sha256"]
            ):
                raise ValueError("node address/hash changed")
        native = np.array([r["native_derivative"] for r in nodes])
        frozen = np.array([r["frozen_derivative"] for r in nodes])
        account = P.finite_account(
            nodes[0]["native"]["candidate_logits"],
            nodes[-1]["native"]["candidate_logits"],
            m["alphabet"]["values"],
            native[0],
        )
        fine, coarse = (
            float(P.integrate(ts, native)),
            float(P.integrate(ts[::2], native[::2])),
        )
        frozen_fine = float(P.integrate(ts, frozen))
        trapezoid = float(np.sum((native[1:] + native[:-1]) * np.diff(ts) / 2))
        rows.append(
            {
                "unit": address["hm24_unit"],
                "family": address["family"],
                "role": address["role"],
                "query": address["query"],
                **account,
                "integral_native": fine,
                "integral_frozen": frozen_fine,
                "integral_readout_increment": fine - frozen_fine,
                "fine_completeness_error": fine - account["native_action"],
                "coarse_completeness_error": coarse - account["native_action"],
                "fine_minus_coarse": fine - coarse,
                "trapezoid_completeness_error": trapezoid - account["native_action"],
                "frozen_completeness_error": frozen_fine
                - account["frozen_readout_logit_action"],
                "derivative_sign_changes": int(np.sum(native[1:] * native[:-1] < 0)),
                "minimum_derivative": float(native.min()),
                "maximum_derivative": float(native.max()),
                "reference_native_frozen_gradient_gap": float(native[0] - frozen[0]),
            }
        )
    summaries = []
    for role in ("all", "source", "validation", "application"):
        for query in sorted({r["query"] for r in rows}):
            chosen = [
                r
                for r in rows
                if r["query"] == query and (role == "all" or r["role"] == role)
            ]
            rms = lambda key, chosen=chosen: float(
                np.sqrt(np.mean([r[key] ** 2 for r in chosen]))
            )
            summaries.append(
                {
                    "role": role,
                    "query": query,
                    "n": len(chosen),
                    **{
                        key + "_rms": rms(key)
                        for key in (
                            "native_action",
                            "full_nonlinear_increment",
                            "decoder_nonlinear_increment",
                            "readout_nonlinear_increment",
                            "fine_completeness_error",
                            "coarse_completeness_error",
                            "frozen_completeness_error",
                        )
                    },
                    "integral_sign_agreements": sum(
                        r["native_action"] * r["integral_native"] > 0 for r in chosen
                    ),
                    "tangent_sign_agreements": sum(
                        r["native_action"] * r["start_tangent_action"] > 0
                        for r in chosen
                    ),
                    "paths_with_derivative_sign_change": sum(
                        r["derivative_sign_changes"] > 0 for r in chosen
                    ),
                }
            )
    (out / "02").mkdir()
    write_csv(out / "02/path_accounts.csv", rows)
    write_csv(out / "02/path_summary.csv", summaries)
    write_json(
        out / "02/summary.json",
        {
            "status": "EVALUATED_FINITE_HM25_NOT_GLOBAL",
            "units": len(rows),
            "nodes": len(rows) * len(ts),
            "calls": read_json(out / "run_status.json"),
            "new_fits": 0,
            "scope": "fixed-decoder path actions, not human personality or LLM-free scoring",
        },
    )
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(read_json(out / "02/summary.json"), flush=True)
    print([r for r in summaries if r["role"] == "application"], flush=True)


def main():
    """Execute one finite stage without mutating prior experiments or results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "check"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    collector_guard(ROOT)
    if args.stage == "check":
        check(args.out)
        print("HM25_FROZEN_FILES_VALID", flush=True)
    else:
        {"prepare": prepare, "run": run, "summarize": summarize}[args.stage](args.out)


if __name__ == "__main__":
    main()

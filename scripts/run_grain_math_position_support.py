#!/usr/bin/env python3
"""HM29 acquire missing token-position support with source-frozen finite actions."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_position_support as P
from grain_research.construction import math_source_field as S
from grain_research.construction import math_span_binding as B
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_position_quotient import check as parent_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_span_binding import (
    array_unit,
    forward,
    hidden_tensor,
    load_observer,
    save_array_unit,
)
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_position_support_v1.json"
PLAN = ROOT / "docs/GRAIN_POSITION_SUPPORT_ACQUISITION_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/position_support_acquisition_r1"
CODE = (
    "grain_research/construction/math_position_support.py",
    "scripts/run_grain_math_position_support.py",
    "scripts/verify_grain_math_position_support.py",
    "tests/test_grain_math_position_support.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def now():
    """Record source-freeze and actual acquisition order in UTC."""
    return datetime.now(timezone.utc).isoformat()


def check(out):
    """Check frozen code, inputs and old results before each bounded stage."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen input changed " + name)
    for name, sha in m["inputs"].items():
        if digest(out / name) != sha:
            raise ValueError("frozen preparation changed " + name)
    return m


def baseline_path(out, unit):
    """Return the exact unit address, not a query-agnostic cache key."""
    return out / "01/baselines" / f"unit_{unit:03d}.json"


def patch_path(out, unit, arm, sign):
    """One atomic artifact per actual finite endpoint call."""
    return out / "01/patches" / f"unit_{unit:03d}_{arm}_{sign}.json"


def prepare(out):
    """Select exact spans by tokens alone and freeze all16 families before scores."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("preserve prepared packet; resume it")
    spec = read_json(SPEC)
    parent = ROOT / spec["quotient_parent"]
    m = parent_check(parent)
    verification = ROOT / spec["quotient_verification"] / "summary.json"
    if read_json(verification)["status"] != "VERIFIED_FINITE_HM28_NOT_GLOBAL":
        raise ValueError("verified HM28 source quotient required")
    files = dict(m["files"])
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("parent artifact changed " + name)
        files[str((parent / name).relative_to(ROOT))] = sha
    files[str(verification.relative_to(ROOT))] = digest(verification)
    qm = read_json(parent / "01/source_manifest.json")
    joint = ROOT / spec["joint_parent"]
    jm = read_json(joint / "01/source_manifest.json")
    refs = read_json(ROOT / spec["reference_packet"] / "00/manifest.json")
    definitions_path = ROOT / spec["query_packet"] / "00/queries.json"
    definitions = read_json(definitions_path)
    files[str(definitions_path.relative_to(ROOT))] = digest(definitions_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    docs, design, attempts = [], [], []
    for family in range(spec["families"]):
        selected = None
        for step in range(len(P.PHRASES)):
            phrase = P.PHRASES[(family + step) % len(P.PHRASES)]
            text = P.material(family, phrase)
            try:
                rendered, bodies, encoded, mask = reference_view(
                    tokenizer, [text], definitions[spec["queries"][0]]["question"]
                )
                qstart = int(np.flatnonzero(mask)[0])
                spans = {
                    name: B.token_span(rendered, bodies[0][0], span, encoded, qstart)
                    for name, span in B.raw_spans(text).items()
                }
                tokens = len(spans["A_property"]["positions"])
                attempts.append({"family": family, "phrase": phrase, "tokens": tokens})
                if tokens == spec["required_property_tokens"]:
                    selected = text, phrase
                    break
            except ValueError as exc:
                attempts.append({"family": family, "phrase": phrase, "error": str(exc)})
        if selected is None:
            raise ValueError(f"no predeclared aligned4-token phrase for family{family}")
        text, phrase = selected
        role = (
            "source" if family < 8 else "validation" if family < 12 else "application"
        )
        docs.append({"family": family, "role": role, "phrase": phrase, "text": text})
        previous = None
        for query in spec["queries"]:
            rendered, bodies, encoded, mask = reference_view(
                tokenizer, [text], definitions[query]["question"]
            )
            qstart = int(np.flatnonzero(mask)[0])
            spans = {
                name: B.token_span(rendered, bodies[0][0], span, encoded, qstart)
                for name, span in B.raw_spans(text).items()
            }
            common = encoded["input_ids"][:qstart], spans
            if previous is not None and previous != common:
                raise ValueError(
                    "material prefix or semantic addresses depend on query"
                )
            previous = common
            if len(spans["A_property"]["positions"]) != 4:
                raise ValueError("property length changed across query")
            design.append(
                {
                    "unit": len(design),
                    "family": family,
                    "role": role,
                    "query": query,
                    "input_ids": encoded["input_ids"],
                    "question_start": qstart,
                    "spans": spans,
                }
            )
    if len(design) != spec["units"] or list(P.ARMS) != spec["arms"]:
        raise ValueError("finite design mismatch")
    arrays = {}
    for query in spec["queries"]:
        entry = qm[query + "/start"]
        with np.load(parent / entry["file"], allow_pickle=False) as data:
            arrays[query + "_old_quotient"] = data["coefficients"].copy()
            arrays[query + "_b"] = P.unit_vector(data["values"][1])
        with np.load(joint / jm[query]["file"], allow_pickle=False) as data:
            arrays[query + "_old_joint"] = data["start_joint"].copy()
        files[str((joint / jm[query]["file"]).relative_to(ROOT))] = digest(
            joint / jm[query]["file"]
        )
    arrays["random_b"] = P.unit_vector(
        np.random.default_rng(spec["random_seed"]).normal(
            size=len(arrays[spec["queries"][0] + "_b"])
        )
    )
    arrays["old_support"] = S.common_covector([np.ones((1, 1)), np.ones((2, 1))])[
        "support"
    ]
    files.update({name: digest(ROOT / name) for name in CODE})
    for path in MODEL.iterdir():
        if path.is_file() and path.suffix in (".json", ".safetensors"):
            files[str(path.relative_to(ROOT))] = digest(path)
    for path in ("01/baselines", "01/patches", "01/source", "00/code"):
        (out / path).mkdir(parents=True, exist_ok=True)
    inputs = {
        "design": design,
        "documents": docs,
        "selection_attempts": attempts,
        "queries": {q: definitions[q] for q in spec["queries"]},
    }
    for name, obj in inputs.items():
        write_json(out / f"00/{name}.json", obj)
    np.savez_compressed(out / "00/directions.npz", **arrays)
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "spec": spec,
            "alphabet": refs["alphabet"],
            "inputs": {f"00/{n}.json": digest(out / f"00/{n}.json") for n in inputs}
            | {"00/directions.npz": digest(out / "00/directions.npz")},
            "frozen_before_outputs": now(),
            "new_material_families": len(docs),
            "span_length_selection_only": True,
            "no_application_fitting": True,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "backward_calls": 0,
            "in_flight": None,
            "stage_seconds": 0.0,
        },
    )
    print(
        {"status": "PREPARED", "units": len(design), "families": len(docs)}, flush=True
    )


def capture(observer, row, m):
    """Differentiate the fixed digit expectation wrt post-block property states."""
    import torch

    slots = {}
    positions = row["spans"]["A_property"]["positions"]
    device = next(observer.model.parameters()).device
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)

    def hook(_module, _args, output):
        original = hidden_tensor(output)
        leaf = original.detach().clone().requires_grad_(True)
        slots["leaf"] = leaf
        slots["h"] = original[0, positions].detach().cpu().numpy().copy()
        slots["q"] = original[0, -1].detach().cpu().numpy().copy()
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    handle = observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(hook)
    try:
        output = observer.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        observer.calls += 1
        logits = output.logits[0, -1]
        candidate = logits[m["alphabet"]["token_ids"]].float()
        values = torch.tensor(
            m["alphabet"]["values"], dtype=torch.float32, device=device
        )
        score = (torch.softmax(candidate, dim=0) * values).sum()
        gradient = torch.autograd.grad(score, slots["leaf"])[0][0, positions]
        arrays = {
            "h": slots["h"],
            "q": slots["q"],
            "gradient": gradient.detach().cpu().numpy().copy(),
        }
        native = native_readout(logits.detach().float().cpu().numpy(), m["alphabet"])
        return arrays, native, float(score.detach().cpu())
    finally:
        handle.remove()


def patched(observer, row, m, original, replacement):
    """Replace only the four property states; audit exact live reference and complement."""
    import torch

    audit = {}
    positions = row["spans"]["A_property"]["positions"]
    device = next(observer.model.parameters()).device

    def hook(_module, _args, output):
        h = hidden_tensor(output)
        current = h[0, positions].detach().cpu().numpy()
        audit["reference_max_error"] = float(np.max(abs(current - original)))
        if audit["reference_max_error"] != 0:
            raise ValueError("live reference changed before patch")
        changed = h.clone()
        changed[0, positions] = torch.tensor(replacement, dtype=h.dtype, device=device)
        outside = torch.ones(h.shape[1], dtype=torch.bool, device=device)
        outside[positions] = False
        audit["other_states_max_error"] = float(
            torch.max(abs(changed[:, outside] - h[:, outside])).cpu()
        )
        if audit["other_states_max_error"] != 0:
            raise ValueError("patch changed outside supplied property span")
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(hook)
    try:
        native = forward(observer, row["input_ids"], m["alphabet"])
    finally:
        handle.remove()
    return native, audit


def recover_in_flight(out, status):
    """Adopt a committed unit only; do not blindly replay an uncertain model call."""
    pending = status["in_flight"]
    if pending is None:
        return status
    path = out / pending["file"]
    if not path.exists():
        raise ValueError(
            "uncertain in-flight call has no committed artifact; explicit recovery needed"
        )
    if pending["backwards"]:
        array_unit(path)
    else:
        read_json(path)
    status = {
        **status,
        "in_flight": None,
        "forward_calls": status["forward_calls"] + 1,
        "backward_calls": status["backward_calls"] + pending["backwards"],
    }
    write_json(out / "run_status.json", status)
    return status


def calls(out, source):
    """Acquire source first or held families later, saving every actual endpoint separately."""
    m = check(out)
    spec = m["spec"]
    status = recover_in_flight(out, read_json(out / "run_status.json"))
    source_manifest = out / "01/source_manifest.json"
    if source and source_manifest.exists():
        raise ValueError("source already frozen; do not reacquire")
    if not source and not source_manifest.exists():
        raise ValueError("source coefficients must be frozen before held observations")
    source_sha = None if source else digest(source_manifest)
    if not source:
        for entry in read_json(source_manifest).values():
            if digest(out / entry["file"]) != entry["sha256"]:
                raise ValueError("source changed before held calls")
    rows = [
        r
        for r in read_json(out / "00/design.json")
        if (r["role"] == "source") == source
    ]
    completed = sum(
        baseline_path(out, r["unit"]).exists()
        for r in read_json(out / "00/design.json")
    )
    endpoints = sum(
        patch_path(out, r["unit"], a, s).exists()
        for r in read_json(out / "00/design.json")
        for a in P.ARMS
        for s in ("plus", "minus")
    )
    if (status["forward_calls"], status["backward_calls"]) != (
        completed + endpoints,
        completed,
    ):
        raise ValueError("call accounting differs from atomic completed artifacts")
    if all(
        baseline_path(out, r["unit"]).exists()
        and all(
            patch_path(out, r["unit"], a, s).exists()
            for a in P.ARMS
            for s in ("plus", "minus")
        )
        for r in rows
    ):
        print("ALREADY_ACQUIRED", flush=True)
        return
    observer = load_observer()
    if any(p.requires_grad for p in observer.model.parameters()):
        raise ValueError("model weights must remain fixed")
    write_json(
        out / ("01/source_observer.json" if source else "01/held_observer.json"),
        observer.manifest(),
    )
    with np.load(out / "00/directions.npz", allow_pickle=False) as data:
        directions = {
            q: P.directions(data[q + "_b"], data["random_b"]) for q in spec["queries"]
        }
    started = time.monotonic()

    def begin(path, backwards):
        nonlocal status
        if (
            status["forward_calls"] >= spec["maximum_forwards"]
            or time.monotonic() - started >= spec["maximum_stage_seconds"]
        ):
            status = {
                **status,
                "status": "RESOURCE_BOUNDARY",
                "stage_seconds": time.monotonic() - started,
            }
            write_json(out / "run_status.json", status)
            return False
        status = {
            **status,
            "status": "SOURCE_CALLS" if source else "HELD_CALLS",
            "in_flight": {
                "file": str(path.relative_to(out)),
                "backwards": backwards,
                "started_at": now(),
            },
        }
        write_json(out / "run_status.json", status)
        return True

    def finish(backwards):
        nonlocal status
        status = {
            **status,
            "in_flight": None,
            "forward_calls": status["forward_calls"] + 1,
            "backward_calls": status["backward_calls"] + backwards,
            "stage_seconds": time.monotonic() - started,
        }
        write_json(out / "run_status.json", status)

    try:
        for row in rows:
            path = baseline_path(out, row["unit"])
            if not path.exists():
                if not begin(path, 1):
                    return
                arrays, native, grad_score = capture(observer, row, m)
                save_array_unit(
                    path,
                    {
                        "address": row,
                        "native": native,
                        "gradient_readout": grad_score,
                        "acquired_at": now(),
                        "source_sha256": source_sha,
                    },
                    arrays,
                )
                finish(1)
            saved = array_unit(path)
            if saved["address"] != row or saved["source_sha256"] != source_sha:
                raise ValueError("baseline address or source freeze changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                h = data["h"].copy()
            for arm in P.ARMS:
                physical = P.physical_patches(
                    h, directions[row["query"]][arm], spec["radius_relative_span_norm"]
                )
                for sign in ("plus", "minus"):
                    path = patch_path(out, row["unit"], arm, sign)
                    address = {"unit": row["unit"], "arm": arm, "sign": sign}
                    if path.exists():
                        old = read_json(path)
                        if (
                            old["address"] != address
                            or old["source_sha256"] != source_sha
                        ):
                            raise ValueError("endpoint address/source changed")
                        continue
                    if not begin(path, 0):
                        return
                    native, audit = patched(observer, row, m, h, physical[sign])
                    write_json(
                        path,
                        {
                            "address": address,
                            "native": native,
                            "audit": audit,
                            "baseline_sha256": digest(baseline_path(out, row["unit"])),
                            "source_sha256": source_sha,
                            "acquired_at": now(),
                        },
                    )
                    finish(0)
            print(
                {
                    "unit": row["unit"],
                    "family": row["family"],
                    "role": row["role"],
                    "forward_calls": status["forward_calls"],
                    "backward_calls": status["backward_calls"],
                },
                flush=True,
            )
        status = {**status, "status": "SOURCE_ACQUIRED" if source else "HELD_ACQUIRED"}
        write_json(out / "run_status.json", status)
    except Exception as exc:
        write_json(
            out / "run_status.json", {**status, "status": "ERROR", "error": repr(exc)}
        )
        raise


def freeze_source(out):
    """Fit only the8 source-family gradients; save rank4 field before held acquisition."""
    m = check(out)
    if (out / "01/source_manifest.json").exists():
        raise ValueError("source already frozen")
    rows = read_json(out / "00/design.json")
    if any(
        baseline_path(out, r["unit"]).exists() for r in rows if r["role"] != "source"
    ):
        raise ValueError("held gradient acquired before source freeze")
    fields = {}
    for q in m["spec"]["queries"]:
        qs = [r for r in rows if r["query"] == q and r["role"] == "source"]
        gradients, hashes = [], {}
        for r in qs:
            p = baseline_path(out, r["unit"])
            array_unit(p)
            hashes[str(p.relative_to(out))] = digest(p)
            hashes[str(p.with_suffix(".npz").relative_to(out))] = digest(
                p.with_suffix(".npz")
            )
            if not all(
                patch_path(out, r["unit"], a, s).exists()
                for a in P.ARMS
                for s in ("plus", "minus")
            ):
                raise ValueError("all declared source controls must be acquired")
            with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
                gradients.append(data["gradient"].astype(float))
        field = S.common_covector(gradients)
        if len(gradients) != 8 or field["rank"] != 4:
            raise ValueError("declared full-rank source required")
        path = out / "01/source" / (q + ".npz")
        np.savez_compressed(path, **field)
        fields[q] = {
            "file": str(path.relative_to(out)),
            "sha256": digest(path),
            "source_hashes": hashes,
            "rank": field["rank"],
            "residual": field["weighted_residual_rms"],
            "frozen_at": now(),
        }
    write_json(out / "01/source_manifest.json", fields)
    status = read_json(out / "run_status.json")
    write_json(out / "run_status.json", {**status, "status": "SOURCE_FROZEN"})
    print({"status": "SOURCE_FROZEN", "fields": len(fields)}, flush=True)


def evaluate(out):
    """Compare actual symmetric decoder actions with old, new and local covectors."""
    m = check(out)
    spec = m["spec"]
    if read_json(out / "run_status.json")["status"] != "HELD_ACQUIRED":
        raise ValueError("finite source and held acquisition must finish first")
    fields = {}
    source = read_json(out / "01/source_manifest.json")
    for q, entry in source.items():
        if digest(out / entry["file"]) != entry["sha256"]:
            raise ValueError("source field changed")
        with np.load(out / entry["file"], allow_pickle=False) as data:
            fields[q] = data["covector"].copy()
    rows, endpoints, residuals = [], [], []
    with np.load(out / "00/directions.npz", allow_pickle=False) as bank:
        for r in read_json(out / "00/design.json"):
            p = baseline_path(out, r["unit"])
            base = array_unit(p)
            with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
                h, g = data["h"].copy(), data["gradient"].astype(float)
            dirs = P.directions(bank[r["query"] + "_b"], bank["random_b"])
            residuals.append(
                {
                    "unit": r["unit"],
                    "query": r["query"],
                    "role": r["role"],
                    "gradient_residual_rms": float(
                        np.sqrt(
                            np.mean((S.moment_map(4).T @ fields[r["query"]] - g) ** 2)
                        )
                    ),
                }
            )
            for arm in P.ARMS:
                physical = P.physical_patches(
                    h, dirs[arm], spec["radius_relative_span_norm"]
                )
                ends = {
                    s: read_json(patch_path(out, r["unit"], arm, s))
                    for s in ("plus", "minus")
                }
                effect = P.contrast(
                    base["native"]["conditional_score"],
                    ends["plus"]["native"]["conditional_score"],
                    ends["minus"]["native"]["conditional_score"],
                )
                moments = S.moment_map(4) @ physical["half_delta"]
                predictions = {
                    name: S.contract(coef, moments)
                    for name, coef in (
                        ("old_joint", bank[r["query"] + "_old_joint"]),
                        ("old_quotient", bank[r["query"] + "_old_quotient"]),
                        ("new_source", fields[r["query"]]),
                    )
                }
                predictions["local_derivative"] = float(
                    np.sum(g * physical["half_delta"])
                )
                rows.append(
                    {
                        "unit": r["unit"],
                        "family": r["family"],
                        "query": r["query"],
                        "role": r["role"],
                        "arm": arm,
                        **effect,
                        **predictions,
                        "physical_delta_norm": float(
                            np.linalg.norm(physical["half_delta"])
                        ),
                        "rounding_known_moment_norm": float(
                            np.linalg.norm(bank["old_support"] @ moments)
                        ),
                        "complement_moment_norm": float(
                            np.linalg.norm((np.eye(4) - bank["old_support"]) @ moments)
                        ),
                    }
                )
                for sign, end in ends.items():
                    if any(end["audit"].values()) or end["baseline_sha256"] != digest(
                        p
                    ):
                        raise ValueError(
                            "endpoint did not retain exact reference/nonpatched states"
                        )
                    endpoints.append(
                        {
                            "unit": r["unit"],
                            "query": r["query"],
                            "role": r["role"],
                            "arm": arm,
                            "sign": sign,
                            **end["native"],
                        }
                    )
    metrics = []
    for q in spec["queries"]:
        for role in ("source", "validation", "application"):
            for arm in P.ARMS:
                metrics.append(
                    {
                        "query": q,
                        "role": role,
                        "arm": arm,
                        **P.summarize(
                            [
                                r
                                for r in rows
                                if (r["query"], r["role"], r["arm"]) == (q, role, arm)
                            ]
                        ),
                    }
                )
    write_csv(out / "02/actions.csv", rows)
    write_csv(out / "02/metrics.csv", metrics)
    write_csv(out / "02/endpoints.csv", endpoints)
    write_csv(out / "02/field_residuals.csv", residuals)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM29_ACQUIRED_NOT_GLOBAL",
            "counts": {
                "references": len(residuals),
                "actions": len(rows),
                "endpoints": len(endpoints),
                "metrics": len(metrics),
                "fields": len(fields),
            },
            "calls": read_json(out / "run_status.json"),
            "source_field": source,
            "scope": spec["scope"],
        },
    )
    paths = [
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.endswith(".tmp")
        and p.name != "run_status.json"
        and p.name != "artifact_hashes.json"
    ]
    write_json(
        out / "02/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    status = read_json(out / "run_status.json")
    write_json(
        out / "run_status.json", {**status, "status": "FINITE_HM29_ACQUIRED_NOT_GLOBAL"}
    )
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Run only the explicitly requested bounded source/held stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "source_calls",
            "freeze_source",
            "held_calls",
            "evaluate",
            "check",
        ),
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        if args.stage == "source_calls":
            calls(args.out, True)
        elif args.stage == "held_calls":
            calls(args.out, False)
        elif args.stage == "check":
            print({"frozen_files": len(check(args.out)["files"])})
        else:
            globals()[args.stage](args.out)


if __name__ == "__main__":
    main()

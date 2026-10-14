#!/usr/bin/env python3
"""HM24: bounded frozen-decoder derivatives and moment-fiber interventions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_decoder_fibers as M
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_output_vector_transport import check as parent_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_span_binding import forward, hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_decoder_fibers_v1.json"
PLAN = ROOT / "docs/GRAIN_DECODER_FIBER_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/decoder_fibers_r1"
CODE = (
    "grain_research/construction/math_decoder_fibers.py",
    "scripts/run_grain_math_decoder_fibers.py",
    "scripts/verify_grain_math_decoder_fibers.py",
    "tests/test_grain_math_decoder_fibers.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out):
    """Validate frozen code and addressed inputs, not an agent's completion claim."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"frozen file changed {name}")
    if digest(out / "00/design.json") != manifest["design_sha256"]:
        raise ValueError("design changed")
    return manifest


def prepare(out):
    """Freeze one output-independent pair per family and all three old queries."""
    if out.exists():
        raise FileExistsError("preserve HM24; resume instead")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent_packet"]
    parent_check(parent, models=True)
    companion = ROOT / spec["parent_companion"] / "summary.json"
    if (
        read_json(companion)["status"]
        != "PHYSICAL_UNITS_VERIFIED_ORIGINAL_RATIO_POINT_FAILURE_RETAINED"
    ):
        raise ValueError("HM23 physical replay required; original failure remains")
    span = ROOT / spec["span_packet"]
    pairs = read_json(span / "00/pairs.json")
    selected = {
        f: min((p for p in pairs if p["family"] == f), key=lambda p: p["pair_id"])
        for f in sorted({p["family"] for p in pairs})
    }
    capture = {(r["doc_id"], r["query"]): r for r in read_json(span / "00/design.json")}
    old = {
        (r["pair_id"], r["query"]): r
        for r in read_json(parent / "00/metadata.json")
        if r["arm"] == "full_property"
    }
    design, files = [], {}
    for pair in selected.values():
        for q in spec["queries"]:
            r = old[pair["pair_id"], q]
            receiver, donor = (
                capture[pair[k], q] for k in ("receiver_doc", "donor_doc")
            )
            ref_npz = r["reference_path"].replace(".json", ".npz")
            donor_npz = str(
                (span / "01/units" / f"unit_{donor['unit']:04d}.npz").relative_to(ROOT)
            )
            design.append(
                {
                    **r,
                    "hm24_unit": len(design),
                    "capture": receiver,
                    "reference_npz": ref_npz,
                    "donor_npz": donor_npz,
                }
            )
            for name in (r["reference_path"], r["unit_path"], ref_npz, donor_npz):
                files[name] = digest(ROOT / name)
    if len(design) != spec["gradient_units"] or len(selected) != spec["families"]:
        raise ValueError("incomplete family subdesign")
    alphabet = read_json(parent / "00/alphabet.json")
    files.update(
        {
            str(p.relative_to(ROOT)): digest(p)
            for p in MODEL.iterdir()
            if p.is_file() and p.suffix in (".json", ".safetensors")
        }
    )
    files.update({name: digest(ROOT / name) for name in CODE})
    files[str(companion.relative_to(ROOT))] = digest(companion)
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        p = out / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "alphabet": alphabet,
            "files": files,
            "design_sha256": digest(out / "00/design.json"),
            "selection_uses_outcomes": False,
            "frozen_before_new_calls": True,
            "original_ratio_point_failure_retained": True,
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "units": 0, "forward_calls": 0, "backward_calls": 0},
    )
    print(
        {
            "status": "PREPARED",
            "units": len(design),
            "maximum_calls": spec["maximum_calls"],
        },
        flush=True,
    )


def replace_hidden(output, h):
    """Preserve tuple metadata when installing an intervention at the same site."""
    return (h, *output[1:]) if isinstance(output, tuple) else h


def run(out):
    """Run finite local gradients and six true patches per addressed reference."""
    import torch

    manifest = check(out)
    spec, alphabet = manifest["spec"], manifest["alphabet"]
    design = read_json(out / "00/design.json")
    done = [
        r
        for r in design
        if (out / "01/units" / f"unit_{r['hm24_unit']:03d}.json").exists()
    ]
    previous = read_json(out / "run_status.json")
    if len(done) == len(design):
        print({"status": "ALREADY_COMPLETE", "units": len(done)}, flush=True)
        return
    observer = load_observer()
    write_json(out / "01/observer.json", observer.manifest())
    start, completed, backward = time.monotonic(), len(done), 0

    def status(label, observer=observer):
        value = {
            "status": label,
            "units": completed,
            "forward_calls": previous["forward_calls"] + observer.calls,
            "backward_calls": previous["backward_calls"] + backward,
            "stage_seconds": time.monotonic() - start,
        }
        write_json(out / "run_status.json", value)
        return value

    status("RUNNING")
    try:
        for row in design:
            path = out / "01/units" / f"unit_{row['hm24_unit']:03d}.json"
            if path.exists():
                if read_json(path)["address"] != row:
                    raise ValueError("resumed address changed")
                continue
            if (
                observer.calls + 7 > spec["maximum_calls"]
                or time.monotonic() - start >= spec["maximum_stage_seconds"]
            ):
                print(status("RESOURCE_BOUNDARY"), flush=True)
                return
            cap = row["capture"]
            names = tuple(cap["spans"])
            with np.load(ROOT / row["reference_npz"]) as p:
                bodies = {name: p["L16_" + name].astype(float) for name in names}
                question = p["L16_Q"].astype(float)
            with np.load(ROOT / row["donor_npz"]) as p:
                donor = p["L16_" + row["span"]].astype(float)
            expected = read_json(ROOT / row["reference_path"])["native"]
            exact_full = (
                read_json(ROOT / row["unit_path"])["native"]["conditional_score"]
                - expected["conditional_score"]
            )
            audit, leaf_slot = {}, []

            def verify_live(
                h, cap=cap, bodies=bodies, names=names, question=question, audit=audit
            ):
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
                    err, float(np.max(abs(h[0, -1].detach().cpu().numpy() - question)))
                )
                audit["live_error"] = max(err, audit.get("live_error", 0.0))
                if err > spec["live_state_tolerance"]:
                    raise ValueError(f"live reference changed {err}")

            def leaf_hook(
                _module, _args, output, verify_live=verify_live, leaf_slot=leaf_slot
            ):
                h = hidden_tensor(output)
                verify_live(h)
                leaf = h.detach().clone().requires_grad_(True)
                leaf_slot.append(leaf)
                return replace_hidden(output, leaf)

            hook = observer.model.model.layers[spec["layer"]].register_forward_hook(
                leaf_hook
            )
            try:
                device = next(observer.model.parameters()).device
                ids = torch.tensor([cap["input_ids"]], dtype=torch.long, device=device)
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
                    gradient = (
                        torch.autograd.grad(score, leaf_slot[0])[0][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
                    backward += 1
                    actual_native = native_readout(
                        logits.detach().cpu().numpy(), alphabet
                    )
            finally:
                hook.remove()
            if actual_native["candidate_logits"] != expected["candidate_logits"]:
                raise ValueError("gradient path baseline changed")
            gradients = {
                name: gradient[cap["spans"][name]["positions"]] for name in names
            }
            del output, leaf_slot, logits, candidate, score, gradient
            split = {
                name: M.split_gradient(
                    gradients[name], 3 if name.endswith("property") else 0
                )
                for name in names
            }
            radius = spec["radius_relative_span_norm"] * np.sqrt(
                sum(np.sum(h**2) for h in bodies.values())
            )
            rng = np.random.default_rng(spec["random_seed"] + row["hm24_unit"])
            random = {
                name: M.split_gradient(
                    rng.normal(size=g.shape), 3 if name.endswith("property") else 0
                )["null"]
                for name, g in gradients.items()
            }
            directions = {
                "visible": M.normalize_direction(
                    {name: split[name]["visible"] for name in names}, radius
                ),
                "null": M.normalize_direction(
                    {name: split[name]["null"] for name in names}, radius
                ),
                "random_null": M.normalize_direction(random, radius),
            }
            arrays = {"g_" + name: g for name, g in gradients.items()}
            records = []
            for arm, direction in directions.items():
                actual_directions, actual_scores = {}, {}
                for sign in (-1, 1):
                    realized = {}

                    def patch_hook(
                        _module,
                        _args,
                        output,
                        sign=sign,
                        names=names,
                        cap=cap,
                        direction=direction,
                        realized=realized,
                        bodies=bodies,
                        verify_live=verify_live,
                    ):
                        h = hidden_tensor(output)
                        verify_live(h)
                        altered = h.clone()
                        for name in names:
                            positions = cap["spans"][name]["positions"]
                            current = h[0, positions]
                            target = current + torch.as_tensor(
                                sign * direction[name], dtype=h.dtype, device=h.device
                            )
                            altered[0, positions] = target
                            realized[name] = (
                                target.detach().cpu().numpy().astype(float)
                                - bodies[name]
                            )
                        return replace_hidden(output, altered)

                    handle = observer.model.model.layers[
                        spec["layer"]
                    ].register_forward_hook(patch_hook)
                    try:
                        native = forward(observer, cap["input_ids"], alphabet)
                    finally:
                        handle.remove()
                    actual_directions[sign], actual_scores[sign] = realized, native
                    for name in names:
                        arrays[f"{arm}_{sign}_{name}"] = realized[name]
                central = (
                    actual_scores[1]["conditional_score"]
                    - actual_scores[-1]["conditional_score"]
                ) / 2
                effective = {
                    name: (actual_directions[1][name] - actual_directions[-1][name]) / 2
                    for name in names
                }
                total = M.directional_action(gradients, effective)
                visible = M.directional_action(
                    {name: split[name]["visible"] for name in names}, effective
                )
                moment_error = max(
                    float(
                        np.max(
                            abs(
                                M.observation_matrix(
                                    len(effective[name]),
                                    3 if name.endswith("property") else 0,
                                )
                                @ effective[name]
                            )
                        )
                    )
                    for name in names
                )
                records.append(
                    {
                        "arm": arm,
                        "central_action": central,
                        "gradient_action": total,
                        "visible_leak_action": visible,
                        "null_gradient_action": total - visible,
                        "moment_change_max": moment_error,
                        "radius": radius,
                        "nonlinear_residual": central - total,
                        "minus_native": actual_scores[-1],
                        "plus_native": actual_scores[1],
                    }
                )
            full_delta = {name: np.zeros_like(g) for name, g in gradients.items()}
            full_delta[row["span"]] = (
                donor.astype(np.float32) - bodies[row["span"]].astype(np.float32)
            ).astype(float)
            full_tangent = M.directional_action(gradients, full_delta)
            pullback = sum(
                float(
                    np.sum(
                        split[name]["moment_covector"]
                        * (
                            M.observation_matrix(
                                len(g), 3 if name.endswith("property") else 0
                            )
                            @ full_delta[name]
                        )
                    )
                )
                for name, g in gradients.items()
            )
            stats = {
                name: {
                    k: v
                    for k, v in split[name].items()
                    if not isinstance(v, np.ndarray)
                }
                for name in names
            }
            np.savez_compressed(path.with_suffix(".npz"), **arrays)
            write_json(
                path,
                {
                    "address": row,
                    "baseline": actual_native,
                    "audit": audit,
                    "span_geometry": stats,
                    "directions": records,
                    "full_native_action": exact_full,
                    "full_tangent_action": full_tangent,
                    "full_moment_pullback_action": pullback,
                    "full_nonlinear_residual": exact_full - full_tangent,
                    "gradient_scope": "six-spans only; outside-state gradients not scored",
                    "gradient_score_dtype": "float32",
                    "patch_readout_dtype": "float64",
                    "arrays_sha256": digest(path.with_suffix(".npz")),
                },
            )
            completed += 1
            print(status("RUNNING"), flush=True)
        print(status("COMPLETE_FINITE_NOT_GLOBAL"), flush=True)
    except BaseException as exc:
        write_json(
            out / "failure_run.json", {"error": repr(exc), "state": status("FAILED")}
        )
        raise
    finally:
        del observer
        torch.mps.empty_cache()


def summarize(out):
    """Summarize every selected unit and retain local gradient/patch disagreements."""
    m = check(out)
    design = read_json(out / "00/design.json")
    directions, units = [], []
    for row in design:
        path = out / "01/units" / f"unit_{row['hm24_unit']:03d}.json"
        v = read_json(path)
        if (
            v["address"] != row
            or digest(path.with_suffix(".npz")) != v["arrays_sha256"]
        ):
            raise ValueError("unit address/hash changed")
        info = {
            k: row[k]
            for k in ("hm24_unit", "family", "role", "query", "span", "tokens")
        }
        units.append(
            {
                **info,
                **{
                    k: v[k]
                    for k in (
                        "full_native_action",
                        "full_tangent_action",
                        "full_moment_pullback_action",
                        "full_nonlinear_residual",
                    )
                },
            }
        )
        for r in v["directions"]:
            directions.append(
                {
                    **info,
                    **{
                        k: val
                        for k, val in r.items()
                        if k not in ("minus_native", "plus_native")
                    },
                }
            )
    (out / "02").mkdir()
    write_csv(out / "02/directions.csv", directions)
    write_csv(out / "02/property_actions.csv", units)
    write_json(
        out / "02/summary.json",
        {
            "status": "EVALUATED_FINITE_HM24_NOT_GLOBAL_COMPLETE",
            "units": len(units),
            "directions": len(directions),
            "source_validation_application": [24, 12, 12],
            "calls": read_json(out / "run_status.json"),
            "spec": m["spec"],
            "new_fits": 0,
            "local_witness_not_personality_validation": True,
        },
    )
    files = {
        str(p.relative_to(out)): digest(p)
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    }
    write_json(out / "02/artifact_hashes.json", files)
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Run one finite resumable stage without touching any historical outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "check"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    collector_guard(ROOT)
    if args.stage == "check":
        check(args.out)
        print("HM24_FROZEN_FILES_VALID", flush=True)
    else:
        {"prepare": prepare, "run": run, "summarize": summarize}[args.stage](args.out)


if __name__ == "__main__":
    main()

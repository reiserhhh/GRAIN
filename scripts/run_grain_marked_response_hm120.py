#!/usr/bin/env python3
"""Freeze and execute a bounded same-evidence marked-response experiment."""

from __future__ import annotations

import argparse
import collections
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_action_expression as A
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_marked_response as M
from grain_research.construction import math_recursive_collector as C
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H

SPEC = ROOT / "configs/grain_marked_response_hm120.json"
CODE = (
    "grain_research/construction/math_marked_response.py",
    "scripts/run_grain_marked_response_hm120.py",
    "tests/test_grain_math_marked_response.py",
    "configs/grain_marked_response_hm120.json",
    "docs/GRAIN_MARKED_RESPONSE_PLAN_HM120.md",
    "grain_research/construction/math_recursive_collector.py",
    "grain_research/construction/math_history_routing.py",
    "suica_core/v7_observations.py",
    "suica_core/v7_operator_controls.py",
    "suica_core/v7_psychometric.py",
)


def save(path, **arrays):
    """Create a numeric artifact without overwriting partial or failed work."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def tags(rows, tokenizer, representation, bundle):
    """Bind three frozen lexical GRAIN slices to the actual stored token slots."""
    import pandas as pd

    from suica_core.suica import TOKEN_RE
    from suica_core.v7_observations import ObservationSpec, canonicalize_comments
    from suica_core.v7_operator_controls import score_operator_from_comments

    raw, binding = [], []
    context_width = max(
        len(tokenizer.encode(r["condition_text"] + "\n", add_special_tokens=False))
        for r in rows
    )
    prefix_length = min(rows[0]["history"])
    for row in rows:
        text = row["condition_text"] + "\n" + row["action_text"] + "\n"
        words = list(TOKEN_RE.finditer(text))
        edges = [
            0,
            words[len(words) // 3].start(),
            words[2 * len(words) // 3].start(),
            len(text),
        ]
        parts = [text[a:b] for a, b in itertools.pairwise(edges)]
        if "".join(parts) != text or min(len(TOKEN_RE.findall(p)) for p in parts) < 12:
            raise ValueError("Every history must have three lossless eligible slices")
        for j, part in enumerate(parts):
            raw.append({"author": f"r{row['id']:04d}s{j}", "body": part})
        slots = []
        for name, origin, char_origin in (
            ("condition_text", prefix_length, 0),
            (
                "action_text",
                prefix_length + context_width,
                len(row["condition_text"]) + 1,
            ),
        ):
            encoded = tokenizer(
                row[name] + "\n", add_special_tokens=False, return_offsets_mapping=True
            )
            for index, (token, (left, right)) in enumerate(
                zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
            ):
                position = origin + index
                if row["input_ids"][position] != token or position in row["padding"]:
                    raise ValueError(
                        "GRAIN character/token address differs from stored history"
                    )
                midpoint = char_origin + (left + right) / 2
                part = min(2, int(np.searchsorted(edges[1:], midpoint, side="right")))
                slots.append((position, part))
        binding.append(slots)
    frame = canonicalize_comments(
        pd.DataFrame(raw), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    units, _, scored = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=representation,
        bundle=bundle,
    )
    if len(scored) != len(raw) or not scored.score_status.eq("SCORED").all():
        raise ValueError("Do not fill refused frozen GRAIN coordinates")
    columns = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
    scores = (
        scored.set_index("user_id")
        .loc[[r["author"] for r in raw], columns]
        .to_numpy(float)
    )
    scores = scores.reshape(len(rows), 3, 4)
    g = np.zeros((len(rows), min(rows[0]["current"]), 4))
    visible = np.zeros(g.shape[:2], bool)
    for r, slots in enumerate(binding):
        for position, part in slots:
            visible[r, position], g[r, position] = True, scores[r, part]
    return g, visible, scores, len(units)


def prepare(out):
    """Seal public source histories and source-frozen marks before differentiation."""
    import joblib
    from transformers import AutoTokenizer

    from suica_core.v7_psychometric import read_factor_bundle

    if out.exists():
        raise FileExistsError("Preserve existing experiment")
    spec = H.load(SPEC)
    src = ROOT / spec["source"]
    old = H.load(src / "00/manifest.json")
    H.verify_hashes(ROOT, old["code"] | old["source"])
    H.verify_hashes(src, old["inputs"])
    H.verify_hashes(src, H.load(src / "01/raw_seal.json"))
    if H.load(src / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Complete frozen history capture required")
    inputs, rows = H.load(src / "02/input.json"), H.load(src / "00/rows.json")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(inputs["weights"]).parent), local_files_only=True
    )
    g, visible, scores, unit_count = tags(
        rows,
        tokenizer,
        joblib.load(ROOT / spec["grain_representation"]),
        read_factor_bundle(ROOT / spec["grain_bundle"]),
    )
    groups = collections.defaultdict(dict)
    for r in rows:
        groups[
            tuple(
                r[name]
                for name in ("variant", "wording", "family", "condition", "owner")
            )
        ][r["polarity"]] = r["id"]
    pairs = [
        {"key": list(key), "low": value[-1], "high": value[1]}
        for key, value in sorted(groups.items())
    ]
    if len(pairs) != spec["pairs"]:
        raise ValueError("All 144 complete pairs required")
    for folder in ("00", "01/units", "02"):
        (out / folder).mkdir(parents=True)
    save(out / "00/marks.npz", g=g, visible=visible, slice_scores=scores)
    H.write(out / "00/pairs.json", pairs)
    H.write(out / "00/rows.json", rows)
    sources = {
        str((src / name).relative_to(ROOT)): H.sha(src / name)
        for name in (
            "00/manifest.json",
            "00/coordinates.npz",
            "00/rows.json",
            "02/input.json",
            "01/raw_seal.json",
            "03/compatibility.json",
            "02_corrected/constructed.npz",
        )
    }
    sources.update(
        {
            name: H.sha(ROOT / name)
            for name in (spec["grain_bundle"], spec["grain_representation"])
        }
    )
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "source": sources,
            "inputs": H.seals(out, out / "00"),
            "input_descriptor": inputs,
            "grain_units": unit_count,
            "mask_changed_pairs": sum(
                inputs["valid"][p["low"]] != inputs["valid"][p["high"]] for p in pairs
            ),
            "scope": spec["scope"],
            "original_scope_complete": False,
        },
    )
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed_pairs": 0,
            "pending": None,
            "seconds": 0.0,
            "attempts": 0,
        },
    )
    print("HM120 prepared", len(pairs), unit_count, "frozen GRAIN slices", flush=True)


def check(out):
    """Verify unchanged code and addressed input artifacts."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    return m


def run(out):
    """Execute bounded path derivatives, checkpointing complete pair batches."""
    import torch

    m, pairs = check(out), H.load(out / "00/pairs.json")
    spec, inputs = m["spec"], m["input_descriptor"]
    status_path = out / "01/status.json"
    status = H.load(status_path)
    if status["state"] not in ("PREPARED", "RUNNING") or status["pending"] is not None:
        raise ValueError("Terminal or uncertain execution is not retried")
    if "transformers" in sys.modules:
        raise RuntimeError("Derivative process must not instantiate a model")
    torch.set_num_threads(2)
    device = spec["device"]
    if C.stream_sha(inputs["weights"]) != inputs["hashes"][inputs["weights"]]:
        raise ValueError("Frozen pretrained maps changed")
    tensor = lambda a: torch.as_tensor(
        np.asarray(a), dtype=torch.float32, device=device
    )
    weights = C.load_weights(inputs["weights"], inputs["blocks"], device)
    if any(w.requires_grad for layer in weights for w in layer.values()):
        raise AssertionError("Pretrained maps must remain frozen")
    with np.load(inputs["coordinates"], allow_pickle=False) as z:
        seeds, cos, sin = (tensor(z[name]) for name in ("seeds", "cos", "sin"))
    bundle = inputs["reader"]
    queries = tensor(
        np.stack(
            [
                R.rotate(
                    np.asarray(bundle["raw_queries"])[q],
                    inputs["endpoint"] + bundle["gaps"][q],
                    bundle["frequencies"],
                    bundle["scaling"],
                )
                for q in range(6)
            ]
        )
    )
    directions = tensor(bundle["directions"])
    nodes, quadrature_weights = M.quadrature(spec["quadrature_nodes"])
    valid = np.asarray(inputs["valid"], float)
    for start in range(status["completed_pairs"], len(pairs), spec["pair_batch"]):
        if status["seconds"] >= spec["max_numeric_seconds"]:
            status["state"] = "BUDGET_STOP"
            write_json(status_path, status)
            raise TimeoutError("Finite numeric budget; no renewal")
        packet = pairs[start : start + spec["pair_batch"]]
        status.update(
            state="RUNNING",
            pending=[start, start + len(packet)],
            attempts=status["attempts"] + 1,
        )
        write_json(status_path, status)
        begun = time.perf_counter()
        try:
            histories = {name: [[], []] for name in ("k", "v")}
            for p in packet:
                for side, name in enumerate(("low", "high")):
                    path = Path(inputs["contexts"][p[name]])
                    if C.stream_sha(path) != inputs["hashes"][str(path.resolve())]:
                        raise ValueError("Frozen history changed")
                    with np.load(path, allow_pickle=False) as z:
                        for channel in histories:
                            histories[channel][side].append(z[channel])
            ends = [
                (
                    tensor(np.stack(histories["k"][side])),
                    tensor(np.stack(histories["v"][side])),
                    tensor(valid[[p[name] for p in packet]]),
                )
                for side, name in enumerate(("low", "high"))
            ]
            delta = tuple(b - a for a, b in zip(*ends, strict=True))
            with torch.no_grad():
                endpoint = [
                    M.read(
                        *M.construct(*x, seeds, cos, sin, weights, inputs["config"]),
                        queries,
                        directions,
                    )
                    .cpu()
                    .numpy()
                    for x in ends
                ]
            contribution = np.zeros((len(packet), 3, 2, 6, valid.shape[1]))
            for t, weight in zip(nodes, quadrature_weights, strict=True):
                x = tuple(
                    (a + float(t) * d).detach().requires_grad_(True)
                    for a, d in zip(ends[0], delta, strict=True)
                )
                reading = M.read(
                    *M.construct(*x, seeds, cos, sin, weights, inputs["config"]),
                    queries,
                    directions,
                )
                focus_difference = reading[:, 0] - reading[:, 1]
                for mode in range(2):
                    for q in range(6):
                        gradients = torch.autograd.grad(
                            focus_difference[:, mode, q].sum(),
                            x,
                            retain_graph=not (mode == 1 and q == 5),
                        )
                        for channel, (grad, change) in enumerate(
                            zip(gradients, delta, strict=True)
                        ):
                            product = grad * change
                            if channel < 2:
                                product = product.sum((1, 3, 4))
                            contribution[:, channel, mode, q] += (
                                weight * product.detach().cpu().numpy()
                            )
                del reading, focus_difference, x, gradients
            seconds = time.perf_counter() - begun
            save(
                out / f"01/units/pairs_{start:04d}.npz",
                attribution=contribution,
                endpoints=np.stack(endpoint),
                ids=np.array([[p["low"], p["high"]] for p in packet]),
            )
            H.write(
                out / f"01/units/pairs_{start:04d}.json",
                {"start": start, "pairs": len(packet), "seconds": seconds},
            )
            status.update(
                completed_pairs=start + len(packet),
                pending=None,
                seconds=status["seconds"] + seconds,
            )
            write_json(status_path, status)
            print(
                "HM120",
                status["completed_pairs"],
                "pairs",
                round(status["seconds"], 3),
                "seconds",
                flush=True,
            )
        except Exception:
            status.update(
                state="FAILED", seconds=status["seconds"] + time.perf_counter() - begun
            )
            write_json(status_path, status)
            raise
    status["state"] = (
        "COMPLETE"
        if status["seconds"] <= spec["max_numeric_seconds"]
        else "BUDGET_STOP"
    )
    write_json(status_path, status)
    H.write(out / "01/seal.json", H.seals(out, out / "01/units"))


def analyze(out):
    """Read every completed response and marked moment without choosing a lens."""
    m, pairs = check(out), H.load(out / "00/pairs.json")
    status = H.load(out / "01/status.json")
    files = sorted((out / "01/units").glob("pairs_*.npz"))
    if not files:
        raise ValueError("No numerical progress to analyze")
    a, endpoint, ids = [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            a.append(z["attribution"])
            endpoint.append(z["endpoints"])
            ids.append(z["ids"])
    a, endpoint, ids = (
        np.concatenate(a),
        np.concatenate(endpoint, axis=1),
        np.concatenate(ids),
    )
    with np.load(out / "00/marks.npz", allow_pickle=False) as z:
        g0, g1 = z["g"][ids[:, 0]], z["g"][ids[:, 1]]
        v0, v1 = z["visible"][ids[:, 0]], z["visible"][ids[:, 1]]
    mask = v0 | v1
    g = (g0 + g1) / (v0.astype(int) + v1.astype(int)).clip(1)[..., None]
    moment = M.marked_moments(a.reshape(len(a), 6, 6, a.shape[-1]), g, mask)
    truth = (endpoint[1, :, 0] - endpoint[1, :, 1]) - (
        endpoint[0, :, 0] - endpoint[0, :, 1]
    )
    recovered = a.sum((1, 4))
    residual = recovered - truth
    with np.load(
        ROOT / m["spec"]["source"] / "02_corrected/constructed.npz", allow_pickle=False
    ) as z:
        source_endpoints = z["readings"][ids.T]
    endpoint_error = float(np.max(abs(source_endpoints - endpoint)))
    maps = {
        "scalar_mass": recovered,
        "first": moment["first"].reshape(len(a), 3, 2, 6, 4).sum(1),
        "connected": moment["connected"].reshape(len(a), 3, 2, 6, 4).sum(1),
        "misaligned_connected": moment["misaligned_connected"]
        .reshape(len(a), 3, 2, 6, 4)
        .sum(1),
    }
    comparisons = []
    if len(a) == len(pairs):
        for name, values in maps.items():
            field = np.zeros((3, 2, 6, 2, 2, *values.shape[1:]))
            for p, value in zip(pairs, values, strict=True):
                v, w, f, c, o = p["key"]
                field[v, w, f, (c + 1) // 2, (o + 1) // 2] = value
            own, odd = (
                field[:, :, :, :, 1],
                (field[:, :, :, :, 1] - field[:, :, :, :, 0]) / 2,
            )
            for component, x in (("own", own), ("actor_odd", odd)):
                for mode in range(2):
                    for v, w in ((1, 0), (1, 1), (2, 0), (2, 1)):
                        s = x[0, 0, :, :, mode].reshape(6, 2, -1)
                        t = x[v, w, :, :, mode].reshape(6, 2, -1)
                        comparison = A.direction_match(s, t)
                        comparisons.append(
                            {
                                "map": name,
                                "component": component,
                                "mode": mode,
                                "variant": v,
                                "wording": w,
                                **comparison,
                            }
                        )
    summary = {
        "status": "COMPLETE_APPROXIMATE_PATH_NOT_FULL"
        if status["state"] == "COMPLETE"
        else "PARTIAL_NOT_FULL",
        "pairs": len(a),
        "numeric_seconds": status["seconds"],
        "new_model_calls": 0,
        "fits": 0,
        "original_HM119_compatibility": "FAILED_RETAINED",
        "endpoint_agreement_max": endpoint_error,
        "completeness_max_abs": float(abs(residual).max()),
        "completeness_RMS": float(np.sqrt(np.mean(residual**2))),
        "response_RMS": float(np.sqrt(np.mean(truth**2))),
        "completeness_RMS_fraction": float(
            np.linalg.norm(residual) / np.linalg.norm(truth)
        ),
        "completeness_no_rescaling_or_refinement": True,
        "channel_signed_RMS": np.sqrt(np.mean(a.sum(-1) ** 2, axis=(0, 2, 3))).tolist(),
        "channel_absolute_mass": abs(a).sum((0, 2, 3, 4)).tolist(),
        "untagged_prompt_absolute_fraction": float(
            (abs(a) * (~mask)[:, None, None, None]).sum() / abs(a).sum()
        ),
        "connected_RMS": float(np.sqrt(np.mean(moment["connected"] ** 2))),
        "misaligned_connected_RMS": float(
            np.sqrt(np.mean(moment["misaligned_connected"] ** 2))
        ),
        "comparisons": comparisons,
        "original_scope_complete": False,
        "scope": m["spec"]["scope"],
    }
    save(
        out / "02/moments.npz",
        attribution=a,
        endpoints=endpoint,
        marks=g,
        mask=mask,
        ids=ids,
        truth=truth,
        residual=residual,
        **moment,
    )
    H.write(out / "02/summary.json", summary)
    H.table(
        out / "02/comparisons.csv",
        [
            {
                key: r.get(key)
                for key in (
                    "map",
                    "component",
                    "mode",
                    "variant",
                    "wording",
                    "status",
                    "matched_mean_cosine",
                    "permutation_mean",
                    "matching_rank_with_ties",
                )
            }
            for r in comparisons
        ],
    )
    print({k: v for k, v in summary.items() if k != "comparisons"}, flush=True)


def main():
    """Keep preparation, derivative execution and numerical analysis separate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.stage](args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""NEXT39: fixed output contrast and native last-attention channel accounting."""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.fixed_seed_attention import decompose_attention
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.native_material_reader import cosine, direction
from scripts.run_grain_multiquery_sensitivity import conditional_torch
from scripts.run_grain_reader_new_author_transport import packet_arrays, verify_hashes
from scripts.run_grain_reader_output_seed import (
    CONTENT,
    NATIVE,
    NEW,
    OLD,
    load_assets,
    summarize,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

QUERIES = ("neuroticism", "confirmation_seeking")
PLAN = ROOT / "docs/GRAIN_FIXED_SEED_ATTENTION_PLAN_V1.md"


def prepare(out):
    """Freeze exact source caches and one identical contrast before native work."""
    units, alphabets, _, files, weight_meta = load_assets()
    if alphabets[QUERIES[0]] != alphabets[QUERIES[1]]:
        raise ValueError("Identical candidate alphabets required")
    v = np.asarray(alphabets[QUERIES[0]]["values"], float)
    c = v - v.mean()
    c /= np.linalg.norm(c)
    jobs = []
    # Old source and new indices are distinct addresses, even when both say 0.
    for cohort, count, directory in (("source4", 4, NATIVE), ("new8", 8, NEW)):
        design = {a["index"]: a for a in read(directory / "private_design.json")}
        for i in range(count):
            inp = (
                CONTENT / f"private_inputs_{i}.npz"
                if cohort == "source4"
                else NEW / "private_inputs" / f"a{i:02d}.npz"
            )
            paths = {
                q: OLD / "private_packets" / f"unit_{i:02d}_{q}.json"
                if cohort == "source4"
                else NEW / "private_prefix" / f"a{i:02d}_{q}.json"
                for q in ("TF_cont", *QUERIES)
            }
            for p in paths.values():
                packet_arrays(p)
                files.extend([p, p.with_suffix(".npz")])
            files += [inp, directory / "private_design.json"]
            for q in QUERIES:
                jobs.append(
                    {
                        "cohort": cohort,
                        "index": i,
                        "query": q,
                        "key": f"{cohort}_a{i:02d}_{q}",
                        "prefix": str(paths[q].relative_to(ROOT)),
                        "common": str(paths["TF_cont"].relative_to(ROOT)),
                        "input": str(inp.relative_to(ROOT)),
                        "state_key": "full_state" if cohort == "source4" else "state",
                        "common_tokens": design[i]["common_prefix_tokens"],
                        "old_score": units[cohort, i, q]["score"],
                        "old_gradient": str(
                            (
                                directory / "private_packets" / f"a{i:02d}_{q}.json"
                            ).relative_to(ROOT)
                        ),
                    }
                )
    from transformers.models.qwen2 import modeling_qwen2

    files += [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/fixed_seed_attention.py",
        ROOT / "grain_research/vector_mechanisms/attention_pullback.py",
        ROOT / "scripts/run_grain_reader_output_seed.py",
        OLD / "observer.json",
    ]
    out.mkdir(exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_jobs.json", jobs)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_FIXED_SEED_ATTENTION_V1",
            "max_calls": 28,
            "max_backwards": 24,
            "max_seconds": 300,
            "c": c.tolist(),
            "alphabet": alphabets[QUERIES[0]],
            "model_sha256": weight_meta["checkpoint_sha256"],
            "implementation_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
            "qwen_implementation_sha256": sha(Path(inspect.getfile(modeling_qwen2))),
            "jobs_sha256": sha(out / "private_jobs.json"),
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "calls": 0,
            "backwards": 0,
            "seconds": 0,
            "completed": [],
        },
    )
    print({"status": "PREPARED", "units": len(jobs), "neural_calls": 0}, flush=True)


def verify(out):
    """Verify prepared inputs without reopening text or fitting targets."""
    from transformers.models.qwen2 import modeling_qwen2

    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    if (
        sha(out / "private_jobs.json") != m["jobs_sha256"]
        or sha(Path(inspect.getfile(modeling_qwen2))) != m["qwen_implementation_sha256"]
    ):
        raise ValueError("Frozen jobs or attention implementation changed")
    return m


def context(job):
    """Load full query state and overwrite only its declared common prefix."""
    _, z = packet_arrays(ROOT / job["prefix"])
    _, common = packet_arrays(ROOT / job["common"])
    full, shared = z[job["state_key"]], common[job["state_key"]]
    n = job["common_tokens"]
    full[:n] = shared[:n]
    with np.load(ROOT / job["input"], allow_pickle=False) as f:
        h, g, pos = f["hidden"], f["grain"], f["positions"]
    if pos.max() >= n or not np.array_equal(full[pos].astype(float), h):
        raise ValueError("Body coordinates do not match canonical prefix")
    return full, h, g, pos


def native_unit(observer, state, kwargs, alphabet, c, pos):
    """One scalar backward with hooks; never replace native attention operations."""
    import torch

    model, capture = observer.model, {}
    layer = model.model.layers[-1]
    attn = layer.self_attn
    if (
        model.config.model_type != "qwen2"
        or attn.sliding_window is not None
        or hasattr(attn, "q_norm")
    ):
        raise ValueError("Unsupported last attention architecture")
    handles = []

    def save_output(name):
        def hook(_module, _args, value):
            capture[name] = value

        return hook

    def save_input(_module, args):
        capture["joined"] = args[0]

    handles.append(layer.input_layernorm.register_forward_hook(save_output("norm")))
    for name in ("q", "k", "v"):
        handles.append(
            getattr(attn, name + "_proj").register_forward_hook(save_output(name))
        )
    handles.append(attn.o_proj.register_forward_pre_hook(save_input))
    leaf = state.detach().requires_grad_(True)
    observer.calls += 1

    def cpu(x):
        return x.detach().float().cpu().numpy()

    try:
        with torch.enable_grad():
            h = leaf
            for block in model.model.layers[17:]:
                h = block(h, **kwargs)
                h = h[0] if isinstance(h, tuple) else h
            logits = model.lm_head(model.model.norm(h[:, -1:]))[0, -1]
            ids = torch.tensor(alphabet["token_ids"], device=state.device)
            contrast = (
                logits[ids] * torch.tensor(c, device=state.device, dtype=state.dtype)
            ).sum()
            score = conditional_torch(logits, alphabet)
            grad, norm_grad, joined_grad = torch.autograd.grad(
                contrast, (leaf, capture["norm"], capture["joined"])
            )
        d, heads, kv = (
            attn.head_dim,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
        )
        cos, sin = kwargs["position_embeddings"]
        mask = kwargs["attention_mask"]
        if mask is not None and not bool((mask[..., -1, :] == 0).all()):
            raise ValueError(
                "Last query has inadmissible keys; do not silently omit mask"
            )
        arrays = {
            "body_gradient": cpu(grad[0, pos]),
            "norm_gradient": cpu(norm_grad[0, pos]),
            "norm_hidden": cpu(capture["norm"][0, pos]),
            "q": cpu(capture["q"][0, -1]).reshape(heads, d),
            "k": cpu(capture["k"][0]).reshape(-1, kv, d),
            "v": cpu(capture["v"][0]).reshape(-1, kv, d),
            "g": cpu(joined_grad[0, -1]).reshape(heads, d),
            "cos": cpu(cos[0]),
            "sin": cpu(sin[0]),
            "joined": cpu(capture["joined"][0, -1]),
        }
        return {
            "score": float(score.detach().cpu()),
            "contrast": float(contrast.detach().cpu()),
        }, arrays
    finally:
        for hook in handles:
            hook.remove()


def collect(out):
    """Run a bounded resumable packet, with explicit IDs and cumulative calls."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, prior = verify(out), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete; zero new neural calls")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Model changed")
    torch.set_num_threads(2)
    loading = time.monotonic()
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    load_seconds = time.monotonic() - loading
    now, old = observer.manifest(), read(OLD / "observer.json")
    for k in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[k] != old[k]:
            raise ValueError("Observer changed: " + k)
    write(out / "observer.json", now)
    attn = observer.model.model.layers[-1].self_attn
    weights = {
        n: getattr(attn, n + "_proj").weight.detach().cpu().numpy() for n in ("k", "v")
    }
    weight_path = out / "private_projections.npz"
    if not weight_path.exists():
        np.savez_compressed(weight_path, **weights)
    else:
        with np.load(weight_path, allow_pickle=False) as z:
            for k, value in weights.items():
                np.testing.assert_array_equal(z[k], value)
    jobs = read(out / "private_jobs.json")
    start, backwards, done = time.monotonic(), 0, list(prior["completed"])

    def status(state):
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "backwards": prior["backwards"] + backwards,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "load_seconds": prior.get("load_seconds", 0) + load_seconds,
            "completed": done.copy(),
            "fits": 0,
            "human_labels_read": False,
        }
        write(out / "run_status.json", value)
        print(
            {k: v for k, v in value.items() if k != "completed"}
            | {"completed_units": len(done)},
            flush=True,
        )
        return value

    try:
        for job in jobs:
            dst = out / "private_packets" / (job["key"] + ".json")
            if job["key"] in done:
                packet_arrays(dst)
                continue
            is_fd = job["cohort"] == "new8" and job["index"] == 0
            if (
                prior["calls"] + observer.calls + 1 + 2 * is_fd > m["max_calls"]
                or prior["backwards"] + backwards + 1 > m["max_backwards"]
                or prior["seconds"] + time.monotonic() - start >= m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            full, h, g, pos = context(job)
            state = torch.tensor(full[None], device="mps")
            kw = cached_suffix_kwargs(observer.model, state)
            backwards += 1
            rec, arr = native_unit(observer, state, kw, m["alphabet"], m["c"], pos)
            rec["score_replay_error"] = abs(rec["score"] - job["old_score"])
            rec["finite_difference"] = None
            if is_fd:
                d, _ = direction(h, g, 0)
                scores = []
                for sign in (1, -1):
                    changed = state.clone()
                    changed[:, pos] += torch.tensor(
                        sign * 0.001 * d[None], device=state.device, dtype=state.dtype
                    )
                    scores.append(
                        float(
                            np.asarray(
                                replay_suffix(observer, changed, kw, m["alphabet"])[
                                    "candidate_logits"
                                ]
                            )
                            @ m["c"]
                        )
                    )
                    del changed
                actual = (scores[0] - scores[1]) / 0.002
                analytic = float(np.sum(d * arr["body_gradient"]))
                rec["finite_difference"] = {
                    "factor": "whole:F01",
                    "epsilon": 0.001,
                    "observed": actual,
                    "analytic": analytic,
                    "finite_delta_error": abs(0.001 * (actual - analytic)),
                }
            np.savez_compressed(dst.with_suffix(".npz"), **arr)
            write(
                dst,
                rec
                | {"key": job["key"], "arrays_sha256": sha(dst.with_suffix(".npz"))},
            )
            done.append(job["key"])
            if rec["score_replay_error"] > 2e-6 or not all(
                np.isfinite(x).all() for x in arr.values()
            ):
                raise ValueError(
                    "Score replay or finite gradient failed; packet preserved"
                )
            status("RUNNING")
            del state, kw, arr
            if len(done) == 2:
                forecast = (time.monotonic() - start) * len(jobs) / 2
                write(
                    out / "resource_forecast.json",
                    {
                        "first_author_seconds": time.monotonic() - start,
                        "forecast_seconds": forecast,
                        "native_cap": m["max_seconds"],
                    },
                )
                if forecast > m["max_seconds"]:
                    status("PAUSED_FORECAST")
                    return
        status("COMPLETE")
    except BaseException:
        status("ERROR")
        raise


def analyze(out, name):
    """Replay formulas from cached heads; report all authors and fixed factors."""
    m = verify(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete fixed native packet required")
    dest = out / name
    dest.mkdir(exist_ok=False)
    jobs = read(out / "private_jobs.json")
    with np.load(out / "private_projections.npz", allow_pickle=False) as z:
        wk, wv = z["k"], z["v"]
    rows, actions, pairs, fields, hashes = [], [], [], {}, {}
    for job in jobs:
        key = job["key"]
        path = out / "private_packets" / (key + ".json")
        rec, a = packet_arrays(path)
        hashes[str(path.relative_to(out))] = sha(path)
        hashes[str(path.with_suffix(".npz").relative_to(out))] = sha(
            path.with_suffix(".npz")
        )
        _, h, g, pos = context(job)
        r = decompose_attention(
            *(a[n] for n in ("q", "k", "v", "g", "cos", "sin")), wk, wv
        )
        v, k = r["value_field"][pos], r["key_field"][pos]
        total = v + k
        norm = a["norm_gradient"]
        error = total - norm
        fields[key] = {
            "H16": a["body_gradient"],
            "value": v,
            "key": k,
            "last_total": total,
        }
        _, old = packet_arrays(ROOT / job["old_gradient"])
        row = {
            "cohort": job["cohort"],
            "index": job["index"],
            "query": job["query"],
            "relative_error": float(np.linalg.norm(error) / np.linalg.norm(norm)),
            "max_abs_error": float(abs(error).max()),
            "output_max_error": float(abs(r["output"] - a["joined"]).max()),
            "value_norm": float(np.linalg.norm(v)),
            "key_norm": float(np.linalg.norm(k)),
            "value_key_cosine": cosine(v, k),
            "total_norm": float(np.linalg.norm(total)),
            "probability_contrast_cosine": cosine(
                old["body_gradient"], a["body_gradient"]
            ),
            "score_replay_error": rec["score_replay_error"],
            "body_attention_mass": float(r["attention"][:, pos].sum(1).mean()),
            "body_negative_selection_fraction": float(
                (r["relative_value"][:, pos] < 0).mean()
            ),
        }
        row["parity_pass"] = (
            row["relative_error"] <= 1e-3 and row["max_abs_error"] <= 1e-5
        )
        rows.append(row)
        for f in range(4):
            dh, _ = direction(h, g, f)
            dn, _ = direction(a["norm_hidden"], g, f)
            val, key_action = float(np.sum(v * dn)), float(np.sum(k * dn))
            actions.append(
                {
                    "cohort": job["cohort"],
                    "index": job["index"],
                    "query": job["query"],
                    "factor": f"whole:F{f + 1:02d}",
                    "h16_action": float(np.sum(a["body_gradient"] * dh)),
                    "value_action": val,
                    "key_action": key_action,
                    "last_total_action": val + key_action,
                    "last_native_action": float(np.sum(norm * dn)),
                    "opposite_channels": bool(val * key_action < 0),
                }
            )
    for cohort, count in (("source4", 4), ("new8", 8)):
        for i in range(count):
            ja, jb = [fields[f"{cohort}_a{i:02d}_{q}"] for q in QUERIES]
            _, oa = packet_arrays(
                ROOT
                / next(
                    j["old_gradient"]
                    for j in jobs
                    if j["cohort"] == cohort
                    and j["index"] == i
                    and j["query"] == QUERIES[0]
                )
            )
            _, ob = packet_arrays(
                ROOT
                / next(
                    j["old_gradient"]
                    for j in jobs
                    if j["cohort"] == cohort
                    and j["index"] == i
                    and j["query"] == QUERIES[1]
                )
            )
            row = {
                "cohort": cohort,
                "index": i,
                "original_probability_cosine": cosine(
                    oa["body_gradient"], ob["body_gradient"]
                ),
            }
            for field in ja:
                row[field + "_cosine"] = cosine(ja[field], jb[field])
                row[field + "_norm_ratio"] = float(
                    np.linalg.norm(ja[field]) / np.linalg.norm(jb[field])
                )
            row["H16_mean_reader_cosine"] = cosine(ja["H16"].mean(0), jb["H16"].mean(0))
            pairs.append(row)
    frame, act, pair = pd.DataFrame(rows), pd.DataFrame(actions), pd.DataFrame(pairs)
    frame.to_csv(dest / "private_units.csv", index=False)
    act.to_csv(dest / "private_actions.csv", index=False)
    pair.to_csv(dest / "private_pairs.csv", index=False)
    summarize(
        frame,
        ["cohort", "query"],
        [c for c in frame if c not in ("cohort", "query", "index")],
    ).to_csv(dest / "channel_geometry.csv", index=False)
    summarize(
        pair, ["cohort"], [c for c in pair if c not in ("cohort", "index")]
    ).to_csv(dest / "query_geometry.csv", index=False)
    summarize(
        act,
        ["cohort", "query", "factor"],
        [c for c in act if c not in ("cohort", "query", "factor", "index")],
    ).to_csv(dest / "grain_actions.csv", index=False)
    summary = {
        "units": len(frame),
        "parity_pass": int(frame.parity_pass.sum()),
        "relative_error_max": float(frame.relative_error.max()),
        "absolute_error_max": float(frame.max_abs_error.max()),
        "score_replay_max": float(frame.score_replay_error.max()),
        "action_closure_max": float(
            abs(act.last_total_action - act.last_native_action).max()
        ),
        "c": m["c"],
        "new_fits": 0,
        "new_neural_calls": 0,
    }
    write(dest / "summary.json", summary)
    write(
        dest / "manifest.json",
        {
            "native_packet_hashes": hashes,
            "projections_sha256": sha(out / "private_projections.npz"),
            "numeric_hashes": {
                p.name: sha(p)
                for p in sorted(dest.glob("*.csv"))
                if not p.name.startswith("._")
            },
        },
    )
    print(summary, flush=True)


def main():
    """Use explicit prepare/collect/analyze stages; refuse to overwrite analyses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--name", default="logic_fixed_seed_attention_v1")
    parser.add_argument("--analysis-name", default="analysis")
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.name, args.analysis_name)):
        raise ValueError("Simple output directory names required")
    out = BASE / args.name
    with threadpool_limits(limits=2):
        if args.stage == "prepare":
            prepare(out)
        elif args.stage == "collect":
            collect(out)
        else:
            analyze(out, args.analysis_name)


if __name__ == "__main__":
    main()

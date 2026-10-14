#!/usr/bin/env python3
"""Connect V1 generator interventions to the reader on original COMPLETE texts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import candidates
from grain_research.vector_mechanisms.reverse_trace import (
    effect_summary,
    equal_norm_random,
)
from scripts.run_grain_persona_generation_trace import (
    BASE,
    append_jsonl,
    capture_o_proj_input,
    load_observer,
    read_jsonl,
    sha,
    write,
)
from scripts.run_grain_persona_reverse_full_context import full_context
from scripts.run_grain_persona_reverse_trace import head_score, read_score


def run(args):
    """Replay V1 scores before suffix-aligned clamps of the fixed reader head."""
    import torch

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source = BASE / "pilot/endpoint_generations.jsonl"
    manifest = {
        "source_sha256": sha(source),
        "script_sha256": sha(Path(__file__)),
        "site": "L12.H0",
        "protocol_sha256": sha(
            ROOT / "docs/GRAIN_PERSONA_REVERSE_FULL_ENDPOINT_PLAN.md"
        ),
    }
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Run identity changed")
    write(out / "manifest.json", manifest)
    old = json.loads((BASE / "pilot/endpoint_local_query.json").read_text())["raw"]
    groups = {}
    for row in read_jsonl(source):
        if row["axis"] == "extraversion":
            groups.setdefault((row["pair"], row["seed"]), {})[row["arm"]] = row
    observer = load_observer(max_tokens=1024, layer_count=2)
    alphabet = candidates("extraversion", observer.tokenizer)
    attn = observer.model.model.layers[12].self_attn
    dim = observer.model.config.hidden_size // observer.model.config.num_attention_heads
    rows = read_jsonl(out / "effects.jsonl")
    done = {r["key"] for r in rows}
    start = time.monotonic()
    replay_error = 0.0
    for (pair, seed), arms in groups.items():
        low_ids, low_pos = full_context(observer, arms["control_low"]["text"])
        low_score = read_score(observer, low_ids, alphabet)
        replay_error = max(
            replay_error,
            abs(
                low_score["score"] - old[f"{pair}|control_low|s{seed}"]["extraversion"]
            ),
        )
        low_state = capture_o_proj_input(observer, attn, low_ids)
        for arm in (
            "intervention_high_into_low",
            "reference_high_prompt",
            "random_equal_norm",
        ):
            ids, pos = full_context(observer, arms[arm]["text"])
            assert [ids[i] for i in pos] == [low_ids[i] for i in low_pos]
            before = read_score(observer, ids, alphabet)
            replay_error = max(
                replay_error,
                abs(before["score"] - old[f"{pair}|{arm}|s{seed}"]["extraversion"]),
            )
            original = capture_o_proj_input(observer, attn, ids)
            mapped = original.clone()
            mapped[0, pos, :] = low_state[0, low_pos, :]
            for kind in ("low_clamp", "self", "random"):
                key = "|".join(map(str, [pair, seed, arm, kind]))
                if key in done:
                    continue
                if time.monotonic() - start > args.max_seconds:
                    write(
                        out / "status.json",
                        {"status": "PARTIAL_BUDGET", "rows": len(rows)},
                    )
                    return
                value = mapped if kind == "low_clamp" else original
                if kind == "random":
                    value = original.clone()
                    x = original[0, pos, :dim].cpu().numpy()
                    y = mapped[0, pos, :dim].cpu().numpy()
                    value[0, pos, :dim] = torch.as_tensor(
                        equal_norm_random(x, y, key),
                        device=value.device,
                        dtype=value.dtype,
                    )
                after = head_score(observer, ids, alphabet, 12, value, pos, 0)
                row = {
                    "key": key,
                    "pair": pair,
                    "seed": seed,
                    "arm": arm,
                    "kind": kind,
                    "site": "L12.H0",
                    "family": arms[arm]["family"],
                    "situation": arms[arm]["situation"],
                    "low": low_score,
                    "before": before,
                    "after": after,
                    "effect": before["score"] - after["score"],
                    "contrast": before["score"] - low_score["score"],
                    "remaining": after["score"] - low_score["score"],
                    "length": len(ids),
                    "low_length": len(low_ids),
                    "positions": pos,
                    "low_positions": low_pos,
                }
                rows.append(row)
                done.add(key)
                append_jsonl(out / "effects.jsonl", row)
        print("full_endpoint", pair, seed, observer.calls, flush=True)
    assert replay_error < 1e-6
    summaries = {}
    for row in rows:
        for task in (row["situation"], "all"):
            key = "|".join([row["arm"], row["kind"], task])
            summaries.setdefault(key, []).append(row)
    write(out / "summary.json", {k: effect_summary(v) for k, v in summaries.items()})
    write(
        out / "status.json",
        {
            "status": "COMPLETE",
            "rows": len(rows),
            "forwards": observer.calls,
            "seconds": time.monotonic() - start,
            "v1_score_replay_max_error": replay_error,
        },
    )


def main():
    """CLI with explicit finite runtime and output identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/persona_reverse_trace_v2/full_endpoint_r1",
    )
    parser.add_argument("--max-seconds", type=int, default=900)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

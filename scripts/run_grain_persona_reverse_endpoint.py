#!/usr/bin/env python3
"""Connect frozen generator interventions to a discovery-selected reader head."""

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
    splice_context,
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
from scripts.run_grain_persona_reverse_trace import head_score, read_score, template


def run(args):
    """Clamp candidate reader heads to low-text states; preserve all generator arms."""
    import torch

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    selected_path = args.reverse_run / "selected_head.json"
    selected = json.loads(selected_path.read_text())["site"]
    sites = sorted({selected, "L16.H7", "L20.H5"})
    source = BASE / "pilot/endpoint_generations.jsonl"
    manifest = {
        "source_sha256": sha(source),
        "selected_sha256": sha(selected_path),
        "script_sha256": sha(Path(__file__)),
        "cap": 96,
        "sites": sites,
        "reverse_runner_sha256": sha(
            ROOT / "scripts/run_grain_persona_reverse_trace.py"
        ),
    }
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Endpoint run identity changed")
    write(out / "manifest.json", manifest)
    groups = {}
    for row in read_jsonl(source):
        if row["axis"] == "extraversion":
            groups.setdefault((row["pair"], row["seed"]), {})[row["arm"]] = row
    observer = load_observer(max_tokens=1024, layer_count=2)
    start = time.monotonic()
    rows = read_jsonl(out / "effects.jsonl")
    done = {r["key"] for r in rows}
    for (pair, seed), arms in groups.items():
        tokens = {
            arm: observer.tokenizer.encode(row["text"], add_special_tokens=False)
            for arm, row in arms.items()
        }
        cap = min(96, *(len(t) for t in tokens.values()))
        for query in ("extraversion", "neuroticism"):
            prefix, suffix = template(observer, query)
            alphabet = candidates(query, observer.tokenizer)
            low_ids, _, pos = splice_context(
                prefix, suffix, tokens["control_low"], tokens["control_low"], cap
            )
            low_score = read_score(observer, low_ids, alphabet)
            downstream = list(range(pos[-1] + 1, len(low_ids)))
            for arm in (
                "intervention_high_into_low",
                "random_equal_norm",
                "reference_high_prompt",
            ):
                if query == "neuroticism" and arm != "intervention_high_into_low":
                    continue
                ids, _, _ = splice_context(
                    prefix, suffix, tokens[arm], tokens["control_low"], cap
                )
                before = read_score(observer, ids, alphabet)
                for site in sites:
                    layer, head = map(
                        int, site.replace("L", "").replace("H", "").split(".")
                    )
                    module = observer.model.model.layers[layer].self_attn
                    donor = capture_o_proj_input(observer, module, low_ids)
                    original = capture_o_proj_input(observer, module, ids)
                    dim = (
                        observer.model.config.hidden_size
                        // observer.model.config.num_attention_heads
                    )
                    for kind in ("low_clamp", "self", "random"):
                        key = "|".join(map(str, [pair, seed, query, arm, site, kind]))
                        if key in done:
                            continue
                        if time.monotonic() - start > args.max_seconds:
                            write(
                                out / "status.json",
                                {"status": "PARTIAL_BUDGET", "rows": len(rows)},
                            )
                            return
                        value = donor if kind == "low_clamp" else original
                        if kind == "random":
                            value = original.clone()
                            slc = slice(head * dim, (head + 1) * dim)
                            x = original[0, downstream, slc].cpu().numpy()
                            y = donor[0, downstream, slc].cpu().numpy()
                            value[0, downstream, slc] = torch.as_tensor(
                                equal_norm_random(x, y, key),
                                device=value.device,
                                dtype=value.dtype,
                            )
                        after = head_score(
                            observer, ids, alphabet, layer, value, downstream, head
                        )
                        source_row = arms[arm]
                        row = {
                            "key": key,
                            "pair": pair,
                            "seed": seed,
                            "query": query,
                            "arm": arm,
                            "site": site,
                            "kind": kind,
                            "family": source_row["family"],
                            "situation": source_row["situation"],
                            "tokens": cap,
                            "low": low_score,
                            "before": before,
                            "after": after,
                            "contrast": before["score"] - low_score["score"],
                            "effect": before["score"] - after["score"],
                            "remaining": after["score"] - low_score["score"],
                        }
                        append_jsonl(out / "effects.jsonl", row)
                        done.add(key)
                        rows.append(row)
            print("endpoint", pair, seed, query, observer.calls, flush=True)
    summaries = {}
    for row in rows:
        key = "|".join(row[k] for k in ("query", "arm", "site", "kind", "situation"))
        summaries.setdefault(key, []).append(row)
    write(out / "summary.json", {k: effect_summary(v) for k, v in summaries.items()})
    write(
        out / "status.json",
        {
            "status": "COMPLETE",
            "rows": len(rows),
            "forwards_this_invocation": observer.calls,
            "seconds": time.monotonic() - start,
        },
    )


def main():
    """Parse explicit bounded output paths; no API or new generator execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reverse-run",
        type=Path,
        default=ROOT / "results/grain_vector_mechanisms/persona_reverse_trace_v2/r1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/persona_reverse_trace_v2/endpoint_r1",
    )
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

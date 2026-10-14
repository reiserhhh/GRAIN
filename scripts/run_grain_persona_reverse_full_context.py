#!/usr/bin/env python3
"""Full articles with token-identical question-suffix head interventions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import (
    SYSTEM,
    candidates,
    question,
    render,
)
from grain_research.vector_mechanisms.reverse_trace import (
    effect_summary,
    equal_norm_random,
    paired_materials,
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
from scripts.run_grain_persona_reverse_trace import head_score, read_score


def full_context(observer, text):
    """Locate a common question suffix without truncating the article."""
    user, _ = render([text], "extraversion")
    ids = observer.prompt_ids(SYSTEM, user)
    start = observer.content_indices(SYSTEM, user, question("extraversion"))[0]
    return ids, list(range(start, len(ids)))


def run(args):
    """Apply frozen heads to complete texts; preserve different absolute positions."""
    import torch

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source = BASE / "expanded/generations.jsonl"
    manifest = {
        "source_sha256": sha(source),
        "script_sha256": sha(Path(__file__)),
        "sites": ["L12.H0", "L16.H7", "L20.H5"],
        "followup_sha256": sha(
            ROOT / "docs/GRAIN_PERSONA_REVERSE_FULL_CONTEXT_FOLLOWUP.md"
        ),
    }
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Run identity changed")
    write(out / "manifest.json", manifest)
    observer = load_observer(max_tokens=1024, layer_count=2)
    alphabet = candidates("extraversion", observer.tokenizer)
    start = time.monotonic()
    rows = read_jsonl(out / "effects.jsonl")
    done = {r["key"] for r in rows}
    for pair in paired_materials(read_jsonl(source)):
        if pair["axis"] != "extraversion":
            continue
        h, ph = full_context(observer, pair["high"]["text"])
        l, pl = full_context(observer, pair["low"]["text"])
        assert [h[i] for i in ph] == [l[i] for i in pl]
        hi, lo = read_score(observer, h, alphabet), read_score(observer, l, alphabet)
        for site in manifest["sites"]:
            layer, head = map(int, site.replace("L", "").replace("H", "").split("."))
            module = observer.model.model.layers[layer].self_attn
            donor = capture_o_proj_input(observer, module, h)
            low = capture_o_proj_input(observer, module, l)
            mapped = low.clone()
            mapped[0, pl, :] = donor[0, ph, :]
            dim = (
                observer.model.config.hidden_size
                // observer.model.config.num_attention_heads
            )
            for kind in ("donor", "self", "random"):
                key = "|".join([pair["pair"], site, kind])
                if key in done:
                    continue
                if time.monotonic() - start > args.max_seconds:
                    write(
                        out / "status.json",
                        {"status": "PARTIAL_BUDGET", "rows": len(rows)},
                    )
                    return
                value = mapped if kind == "donor" else low
                if kind == "random":
                    value = low.clone()
                    slc = slice(head * dim, (head + 1) * dim)
                    value[0, pl, slc] = torch.as_tensor(
                        equal_norm_random(
                            low[0, pl, slc].cpu().numpy(),
                            mapped[0, pl, slc].cpu().numpy(),
                            key,
                        ),
                        device=value.device,
                        dtype=value.dtype,
                    )
                result = head_score(observer, l, alphabet, layer, value, pl, head)
                row = {k: v for k, v in pair.items() if k not in ("high", "low")}
                row.update(
                    key=key,
                    site=site,
                    kind=kind,
                    low=lo,
                    high=hi,
                    result=result,
                    effect=result["score"] - lo["score"],
                    contrast=hi["score"] - lo["score"],
                    high_length=len(h),
                    low_length=len(l),
                    high_suffix=ph,
                    low_suffix=pl,
                    suffix_ids=[h[i] for i in ph],
                )
                append_jsonl(out / "effects.jsonl", row)
                rows.append(row)
                done.add(key)
        print("full_context", pair["pair"], observer.calls, flush=True)
    groups = {}
    for row in rows:
        key = "|".join(row[k] for k in ("site", "kind", "situation"))
        groups.setdefault(key, []).append(row)
    write(out / "summary.json", {k: effect_summary(v) for k, v in groups.items()})
    write(
        out / "status.json",
        {
            "status": "COMPLETE",
            "rows": len(rows),
            "forwards": observer.calls,
            "seconds": time.monotonic() - start,
        },
    )


def main():
    """CLI for a single bounded full-context application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/persona_reverse_trace_v2/full_context_r1",
    )
    parser.add_argument("--max-seconds", type=int, default=900)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

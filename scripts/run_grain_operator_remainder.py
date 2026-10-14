#!/usr/bin/env python3
"""One finite remainder action, completing an exact four-corner accounting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_operator import metrics
from grain_research.vector_mechanisms.direct_traits import candidates
from scripts.run_grain_conditional_operator import QUERIES, inputs, table
from scripts.run_grain_conditional_operator_native import cache_path, context, intervene
from scripts.run_grain_persona_generation_trace import (
    append_jsonl,
    load_observer,
    read_jsonl,
    sha,
    write,
)


def account(a, r, d):
    """Compute interaction and symmetric finite contributions without ratios."""
    interaction = d - a - r
    return interaction, a + interaction / 2, r + interaction / 2


def run(out):
    """Add the single missing action to all existing application pairs."""
    manifest = {
        "script": sha(Path(__file__)),
        "protocol": sha(ROOT / "docs/GRAIN_OPERATOR_REMAINDER_FOLLOWUP_20260915.md"),
        "actions": sha(out / "action_arrays.npz"),
        "original_effects": sha(out / "native_effects.jsonl"),
    }
    mp = out / "remainder_manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError("Remainder source changed")
    write(mp, manifest)
    _, pairs = inputs()
    with np.load(out / "action_arrays.npz") as z:
        remainder = z["dh"] - z["predicted"]
    originals = {
        (r["pair"], r["query"], r["kind"]): r
        for r in read_jsonl(out / "native_effects.jsonl")
    }
    records = read_jsonl(out / "remainder_effects.jsonl")
    done = {r["key"] for r in records}
    observer = load_observer(max_tokens=1024, layer_count=2)
    alphabets = {q: candidates(q, observer.tokenizer) for q in QUERIES}
    start = time.monotonic()
    for i, p in enumerate(pairs):
        if p["split"] == "source":
            continue
        offset = 0
        for q in QUERIES:
            with np.load(cache_path(out, p["low"]["text_id"], q)) as z:
                shape = z["head"].shape
            size = int(np.prod(shape))
            action = remainder[i, offset : offset + size].reshape(shape)
            offset += size
            key = f"{p['pair']}|{q}"
            if key in done:
                continue
            ids, pos = context(observer, p["low"]["text"], q)
            native = intervene(observer, ids, pos, alphabets[q], action)
            compiled = originals[(p["pair"], q, "compiled")]
            donor = originals[(p["pair"], q, "donor")]
            r_effect = native["score"] - compiled["low"]
            interaction, a_share, r_share = account(
                compiled["effect"], r_effect, donor["effect"]
            )
            record = {
                k: p[k]
                for k in ("pair", "axis", "family", "situation", "seed", "split")
            }
            record.update(
                key=key,
                query=q,
                low=compiled["low"],
                residual_score=native["score"],
                residual=r_effect,
                compiled=compiled["effect"],
                donor=donor["effect"],
                interaction=interaction,
                symmetric_compiled=a_share,
                symmetric_remainder=r_share,
                mass=native["mass"],
            )
            append_jsonl(out / "remainder_effects.jsonl", record)
            records.append(record)
            done.add(key)
        if len(records) % 24 == 0:
            print("remainder", len(records), flush=True)
    summaries = []
    for q in QUERIES:
        for axis in ("all", *QUERIES):
            rr = [
                r
                for r in records
                if r["query"] == q and (axis == "all" or r["axis"] == axis)
            ]
            for k in (
                "compiled",
                "residual",
                "interaction",
                "symmetric_compiled",
                "symmetric_remainder",
            ):
                v = np.array([r[k] for r in rr])
                summaries.append(
                    {
                        "query": q,
                        "axis": axis,
                        "component": k,
                        "rms": float(np.sqrt(np.mean(v**2))),
                        **metrics(v, [r["donor"] for r in rr]),
                    }
                )
    assert len(records) == 144
    error = max(
        abs(r["compiled"] + r["residual"] + r["interaction"] - r["donor"])
        for r in records
    )
    assert error < 1e-12
    table(out / "remainder_summary.csv", summaries)
    write(
        out / "remainder_status.json",
        {
            "status": "COMPLETE",
            "rows": len(records),
            "forwards_this_invocation": observer.calls,
            "seconds": time.monotonic() - start,
            "identity_error": error,
        },
    )
    print(json.dumps([s for s in summaries if s["axis"] == "all"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    run(ap.parse_args().output)

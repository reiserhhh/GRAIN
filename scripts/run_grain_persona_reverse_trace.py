#!/usr/bin/env python3
"""Bounded article-reader localization, head decomposition, and endpoint connection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import (
    SYSTEM,
    candidates,
    native_readout,
    render,
)
from grain_research.vector_mechanisms.reverse_trace import (
    effect_summary,
    equal_norm_random,
    paired_materials,
    select_discovery,
    splice_context,
)
from grain_research.vector_mechanisms.trace import Patch
from scripts.run_grain_persona_generation_trace import (
    BASE,
    append_jsonl,
    capture_o_proj_input,
    load_observer,
    make_row_replace,
    read_jsonl,
    sha,
    write,
)


def template(observer, query):
    """Split a frozen reader template at an explicitly replaceable article marker."""
    marker = "REPLACEABLEARTICLEMARKER"
    user, _ = render([marker], query)
    ids = observer.prompt_ids(SYSTEM, user)
    positions = observer.content_indices(SYSTEM, user, marker)
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("Noncontiguous marker")
    return ids[: positions[0]], ids[positions[-1] + 1 :]


def contexts(observer, high, low, query):
    """Build paired reader prompts with no generator prompt or condition metadata."""
    prefix, suffix = template(observer, query)
    h = observer.tokenizer.encode(high, add_special_tokens=False)
    l = observer.tokenizer.encode(low, add_special_tokens=False)
    return splice_context(prefix, suffix, h, l)


def read_score(observer, ids, alphabet, patch=None):
    """Read fixed-alphabet logits, retaining compliance mass separately."""
    _, logits = observer.capture(ids, positions=[len(ids) - 1], patch=patch)
    out = native_readout(logits, alphabet)
    return {"score": out["conditional_score"], "mass": out["candidate_mass"]}


def head_score(observer, ids, alphabet, layer, donor, positions, head=None):
    """Patch only downstream head output rows and remove the hook on every exit."""
    config = observer.model.config
    dim = config.hidden_size // config.num_attention_heads
    module = observer.model.model.layers[layer].self_attn.o_proj
    handle = module.register_forward_pre_hook(
        make_row_replace(donor, positions, head, dim)
    )
    try:
        return read_score(observer, ids, alphabet)
    finally:
        handle.remove()


def run(args):
    """Execute one finite, resumable packet; never modify V1 sources."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source = BASE / "expanded/generations.jsonl"
    specification = {
        "source_sha256": sha(source),
        "cap": 128,
        "smoke": args.smoke,
        "sites": [0, 4, 8, 12, 16, 20, 24],
        "head_layers": [12, 16, 20],
        "runner_sha256": sha(Path(__file__)),
        "helper_sha256": sha(
            ROOT / "grain_research/vector_mechanisms/reverse_trace.py"
        ),
        "protocol_sha256": sha(ROOT / "docs/GRAIN_PERSONA_REVERSE_TRACE_PLAN_V2.md"),
    }
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != specification:
        raise ValueError("Frozen run changed; choose a new output directory")
    write(manifest, specification)
    pairs = paired_materials(read_jsonl(source))
    if args.smoke:
        pairs = pairs[:1]
    observer = load_observer(max_tokens=1024)
    observer.sites = {
        f"block_{i:02d}": observer.model.model.layers[i] for i in specification["sites"]
    }
    write(out / "model_manifest.json", observer.manifest())
    start = time.monotonic()
    rows = read_jsonl(out / "effects.jsonl")
    done = {r["key"] for r in rows}

    def save(base, stage, site, kind, result, reference, contrast):
        key = "|".join(map(str, [base["pair"], base["query"], stage, site, kind]))
        if key in done:
            return
        row = {
            **base,
            "key": key,
            "stage": stage,
            "site": site,
            "kind": kind,
            **result,
            "effect": result["score"] - reference,
            "contrast": contrast,
        }
        append_jsonl(out / "effects.jsonl", row)
        rows.append(row)
        done.add(key)

    def needed(base, stage, site, kind):
        if time.monotonic() - start > args.max_seconds:
            raise TimeoutError("Bounded packet time limit")
        return (
            "|".join(map(str, [base["pair"], base["query"], stage, site, kind]))
            not in done
        )

    basics = {}
    for pair in pairs:
        for query in ("extraversion", "neuroticism"):
            base = {k: v for k, v in pair.items() if k not in ("high", "low")}
            base["query"] = query
            alphabet = candidates(query, observer.tokenizer)
            h, l, positions = contexts(
                observer, pair["high"]["text"], pair["low"]["text"], query
            )
            dh, lh = observer.capture(h, positions=positions)
            dl, ll = observer.capture(l, positions=positions)
            sh, sl = native_readout(lh, alphabet), native_readout(ll, alphabet)
            hi, lo = sh["conditional_score"], sl["conditional_score"]
            contrast = hi - lo
            basics[(pair["pair"], query)] = (base, h, l, positions, hi, lo, alphabet)
            address = {
                "pair": pair["pair"],
                "query": query,
                "high_id": pair["high"]["text_id"],
                "low_id": pair["low"]["text_id"],
                "high_prompt_ids": h,
                "low_prompt_ids": l,
                "article_positions": positions,
                "note": "ordinal prefix correspondence, not semantic alignment",
            }
            write(
                out
                / "addresses"
                / (pair["pair"].replace("|", "_") + "_" + query + ".json"),
                address,
            )
            for kind, result in [("high", sh), ("low", sl)]:
                save(
                    base,
                    "baseline",
                    "prefix",
                    kind,
                    {
                        "score": result["conditional_score"],
                        "mass": result["candidate_mass"],
                    },
                    lo,
                    contrast,
                )
            for pole in ("high", "low"):
                if needed(base, "baseline", "full", pole):
                    user, _ = render([pair[pole]["text"]], query)
                    result = read_score(
                        observer, observer.prompt_ids(SYSTEM, user), alphabet
                    )
                    save(base, "baseline", "full", pole, result, lo, contrast)
            for site in observer.sites:
                for kind in ("donor", "random", "self"):
                    if kind == "self" and site != "block_12":
                        continue
                    if not needed(base, "residual", site, kind):
                        continue
                    donor = dh[site] if kind == "donor" else dl[site]
                    if kind == "random":
                        donor = equal_norm_random(
                            dl[site], dh[site], base["pair"] + query + site
                        )
                    result = read_score(
                        observer, l, alphabet, Patch(site, positions, donor)
                    )
                    save(base, "residual", site, kind, result, lo, contrast)
            print("residual", base["pair"], query, observer.calls, flush=True)
    selected = select_discovery([r for r in rows if r["stage"] == "residual"], "site")
    write(
        out / "selected_residual.json",
        {"site": selected, "criterion": "E opinion donor MSE only"},
    )

    for base, h, l, positions, hi, lo, alphabet in basics.values():
        contrast = hi - lo
        dh, _ = observer.capture(h, positions=positions)
        for quarter, indices in enumerate(np.array_split(np.arange(len(positions)), 4)):
            site = f"{selected}/quarter{quarter}"
            if needed(base, "quarters", site, "donor"):
                pos = [positions[i] for i in indices]
                result = read_score(
                    observer, l, alphabet, Patch(selected, pos, dh[selected][indices])
                )
                save(base, "quarters", site, "donor", result, lo, contrast)
        if base["axis"] != "extraversion" or base["query"] != "extraversion":
            continue
        suffix = list(range(positions[-1] + 1, len(l)))
        for layer in specification["head_layers"]:
            attn = observer.model.model.layers[layer].self_attn
            donor = capture_o_proj_input(observer, attn, h)
            low = capture_o_proj_input(observer, attn, l)
            dim = (
                observer.model.config.hidden_size
                // observer.model.config.num_attention_heads
            )
            for head in (None, *range(observer.model.config.num_attention_heads)):
                site = f"L{layer}.H{head}" if head is not None else f"L{layer}.all"
                if needed(base, "heads", site, "donor"):
                    result = head_score(
                        observer, l, alphabet, layer, donor, suffix, head
                    )
                    save(base, "heads", site, "donor", result, lo, contrast)
                if head is None or (layer, head) in ((16, 7), (20, 5)):
                    for kind in ("self", "random"):
                        if not needed(base, "heads", site, kind):
                            continue
                        value = low.clone()
                        if kind == "random":
                            slc = (
                                slice(None)
                                if head is None
                                else slice(head * dim, (head + 1) * dim)
                            )
                            x = low[0, suffix, slc].cpu().numpy()
                            y = donor[0, suffix, slc].cpu().numpy()
                            import torch

                            value[0, suffix, slc] = torch.as_tensor(
                                equal_norm_random(x, y, base["pair"] + site),
                                device=value.device,
                                dtype=value.dtype,
                            )
                        result = head_score(
                            observer, l, alphabet, layer, value, suffix, head
                        )
                        save(base, "heads", site, kind, result, lo, contrast)
        print("quarters/heads", base["pair"], observer.calls, flush=True)

    head_rows = [r for r in rows if r["stage"] == "heads" and ".all" not in r["site"]]
    selected_head = select_discovery(head_rows, "site")
    write(
        out / "selected_head.json",
        {"site": selected_head, "criterion": "E opinion donor MSE only"},
    )
    summarize(out, rows, selected, selected_head)
    write(
        out / "status.json",
        {
            "status": "COMPLETE",
            "pairs": len(pairs),
            "rows": len(rows),
            "forwards_this_invocation": observer.calls,
            "seconds": time.monotonic() - start,
        },
    )


def summarize(out, rows, site, head):
    """Save grouped effects and a readable finite-packet report without pooling axes."""
    groups = {}
    for r in rows:
        key = "|".join(
            str(r[k]) for k in ("stage", "axis", "query", "situation", "site", "kind")
        )
        groups.setdefault(key, []).append(r)
    write(out / "summary.json", {k: effect_summary(v) for k, v in groups.items()})
    lines = [
        "# Article-to-trait reverse trace V2",
        "",
        f"Selected on opinion only: residual `{site}`, reader head `{head}`.",
        "",
        "All effects below are normalized 0-1 reader-score changes, not accuracy.",
        "Inputs are matched token prefixes; seeds stay together in family/task bootstrap.",
        "",
        "| Stage/site | Generated axis | Query | Task | n | Contrast | Effect [95%] | Slope |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for key, values in groups.items():
        r = values[0]
        if r["kind"] != "donor" or r["site"] not in (site, head, "L16.H7", "L20.H5"):
            continue
        s = effect_summary(values)
        low, high = s["mean_effect_ci"]
        slope = "NA" if s["slope"] is None else f"{s['slope']:.3f}"
        lines.append(
            f"| {r['stage']}/{r['site']} | {r['axis']} | {r['query']} | {r['situation']} | {s['n']} | {s['mean_contrast']:.4f} | {s['mean_effect']:.4f} [{low:.4f}, {high:.4f}] | {slope} |"
        )
    lines += [
        "",
        "Full-text scores and candidate mass are in effects.jsonl; do not equate",
        "prefix effects with the previous full-text AUC. Quarter patches carry",
        "contextual states, not isolated lexical causes. Head matches do not",
        "establish an exact inverse or personality-specific neurons. Endpoint",
        "connection is a separately recorded follow-up, not implied by this scan.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    """CLI with explicit output identity and a wall-clock stop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_vector_mechanisms/persona_reverse_trace_v2/r1",
    )
    parser.add_argument("--max-seconds", type=int, default=2700)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        write(
            args.output / "status.json",
            {
                "status": "PARTIAL_BUDGET"
                if isinstance(exc, TimeoutError)
                else "ERROR",
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()

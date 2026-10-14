#!/usr/bin/env python3
"""Stages of the persona generation trace: generate | read | judge | localize | endpoint.

generate: local open checkpoint writes free text under each frozen system prompt.
read:     blinded readers see only the text (TF-IDF probe, frozen GRAIN 4-slice
          coordinates + probe, local text-only trait question readout).
judge:    DeepSeek Flash blinded evaluator (separate branch, never used for selection).
localize: teacher-forced, position-aligned residual/MLP/attention patching with
          random equal-norm and self-patch controls on discovery/development pairs.
endpoint: regenerate short continuations under the selected intervention and
          score frozen GRAIN coordinates (delta_g), kept separate from delta log p.
Every stage checkpoints incrementally and respects finite budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.persona_trace import (
    audit_text,
    strip_leak,
)

BASE = ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1"
MODEL_DIR = ROOT / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ENV = ROOT.parent / "project persona" / ".env"
JUDGE_SYSTEM = (
    "You are a blind rater of short texts. You see only the text. Rate the writer's "
    "apparent extraversion (sociability, assertiveness, energetic engagement with people) "
    "and neuroticism (tendency to worry and to experience negative emotion) on integer "
    "scales from 0 (very low) to 9 (very high). Also report whether the text explicitly "
    "names personality traits and whether it stays on the requested topic. Return JSON only "
    'with keys: extraversion, neuroticism, explicit_trait_labels (true/false), on_topic (true/false).'
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=float) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=float) + "\n")


def load_observer(max_tokens=1024, layer_count=28):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no automatic remote fallback")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(str(MODEL_DIR), device="mps", dtype="float32", max_tokens=max_tokens, local_files_only=True)
    observer.model.requires_grad_(False)
    decoder = observer.model.model
    n = len(decoder.layers)
    keep = sorted(set(np.linspace(0, n - 1, min(layer_count, n), dtype=int).tolist()))
    observer.sites = {f"block_{i:02d}": decoder.layers[i] for i in keep}
    return observer


# ---------------------------------------------------------------- generate
def stage_generate(run, args):
    import torch

    manifest = json.loads((run / "manifest.json").read_text())
    rows = json.loads((run / "materials.json").read_text())
    gen = manifest["generation"]
    done = {r["text_id"] for r in read_jsonl(run / "generations.jsonl")}
    todo = [r for r in rows if r["text_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    observer = load_observer(max_tokens=768)
    write(run / "model_manifest.json", observer.manifest() | {"weights_sha256_checked": sha(MODEL_DIR / "model.safetensors") == manifest["model_weights_sha256_pinned"]})
    start = time.monotonic()
    eos = observer.tokenizer.eos_token_id
    for i, row in enumerate(todo):
        if time.monotonic() - start > args.max_seconds:
            write(run / "generate_status.json", {"status": "PARTIAL_BUDGET", "done": len(done) + i, "total": len(rows)})
            print("budget stop", flush=True)
            return
        ids = observer.prompt_ids(row["system"], row["user"])
        span = observer.content_indices(row["system"], row["user"], row["persona"])
        torch.manual_seed(row["seed"])
        device = next(observer.model.parameters()).device
        tokens = torch.tensor([ids], device=device)
        with torch.inference_mode():
            out = observer.model.generate(
                input_ids=tokens, attention_mask=torch.ones_like(tokens), do_sample=gen["do_sample"],
                temperature=gen["temperature"], top_p=gen["top_p"], max_new_tokens=gen["max_new_tokens"],
                pad_token_id=eos, eos_token_id=eos, use_cache=True,
            )
        new = out[0, len(ids):].tolist()
        finish = "eos" if (eos in new) else "length"
        new_ids = new[: new.index(eos)] if eos in new else new
        text = observer.tokenizer.decode(new_ids, skip_special_tokens=True)
        record = {**row, "prompt_ids": ids, "persona_span": span, "new_ids": new_ids, "text": text,
                  "finish_reason": finish, "n_new": len(new_ids), "audit": audit_text(text, row["situation"], finish),
                  "seconds": time.monotonic() - start}
        append_jsonl(run / "generations.jsonl", record)
        print(f"{len(done) + i + 1}/{len(rows)} {row['text_id']} n_new={len(new_ids)} finish={finish} leak={record['audit']['leak_hits']}", flush=True)
    write(run / "generate_status.json", {"status": "COMPLETE", "total": len(rows), "seconds": time.monotonic() - start, "generation": gen})


# ---------------------------------------------------------------- read
def grain_coordinates(texts):
    """Frozen 4-slice GRAIN coordinates on raw text; None when the minimum text is unmet."""
    import joblib

    from scripts.run_grain_explanation_chain import slice_scores
    from scripts.run_grain_native_material_reader import BUNDLE, REPRESENTATION
    from suica_core.v7_psychometric import read_factor_bundle

    representation, bundle = joblib.load(REPRESENTATION), read_factor_bundle(BUNDLE)
    out = {}
    for key, text in texts.items():
        try:
            _, scores, err = slice_scores(text, representation, bundle)
            out[key] = {"g": scores.mean(axis=0).tolist(), "slices": scores.tolist(), "replay_error": err}
        except ValueError as exc:
            out[key] = {"g": None, "reason": str(exc)[:80]}
    return out


def local_query_readout(observer, texts, tasks):
    from grain_research.vector_mechanisms.direct_traits import (
        SYSTEM,
        candidates,
        native_readout,
        render,
    )

    out = {}
    for key, text in texts.items():
        out[key] = {}
        for task in tasks:
            user, _ = render([text], task)
            ids = observer.prompt_ids(SYSTEM, user)
            _, logits = observer.capture(ids)
            out[key][task] = native_readout(logits, candidates(task, observer.tokenizer))["conditional_score"]
    return out


def stage_read(run, args):
    gens = read_jsonl(run / "generations.jsonl")
    texts = {g["text_id"]: g["text"] for g in gens}
    stripped = {k: strip_leak(v) for k, v in texts.items()}
    readout = {"grain": grain_coordinates(texts), "grain_stripped": grain_coordinates(stripped)}
    write(run / "readout_grain.json", readout)
    observer = load_observer(max_tokens=1024, layer_count=2)
    start = time.monotonic()
    local = {"raw": local_query_readout(observer, texts, ("extraversion", "neuroticism")),
             "stripped": local_query_readout(observer, stripped, ("extraversion", "neuroticism"))}
    write(run / "readout_local_query.json", local | {"forwards": observer.calls, "seconds": time.monotonic() - start,
                                                     "context": "article text + fixed question only; no persona prompt, condition id or cache"})
    print(json.dumps({"grain_available": sum(v["g"] is not None for v in readout["grain"].values()), "texts": len(texts), "local_forwards": observer.calls}))


# ---------------------------------------------------------------- judge (DeepSeek Flash)
def load_env():
    values = {}
    for line in ENV.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def deepseek_request(base_url, key, model, messages, *, max_tokens, timeout, retries):
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}}
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return data, attempt
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {getattr(exc, 'code', '')}"
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek request failed after retries: {last}")


def stage_judge(run, args):
    env = load_env()
    key, base = env.get("DEEPSEEK_API_KEY"), env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in .env (names only inspected)")
    model = args.judge_model
    gens = read_jsonl(run / "generations.jsonl")
    done = {r["text_id"] for r in read_jsonl(run / "judge_deepseek.jsonl") if "rating" in r and "parse_error" not in r["rating"]}
    acct_path = run / "api_accounting.json"
    acct = json.loads(acct_path.read_text()) if acct_path.exists() else {"model_requested": model, "requests": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0, "failures": 0, "models_reported": {}}
    acct["limits"] = {"max_requests": args.max_requests, "max_total_tokens": args.max_total_tokens, "note": "limits of the latest invocation; counts are cumulative over invocations for this run directory"}
    todo = [g for g in gens if g["text_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    for g in todo:
        total = acct["prompt_tokens"] + acct["completion_tokens"]
        if acct["requests"] >= args.max_requests or total >= args.max_total_tokens:
            write(acct_path, acct | {"status": "BUDGET_STOP"})
            print("judge budget stop", flush=True)
            return
        messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "Text:\n" + g["text"] + "\n\nReturn the JSON now."}]
        try:
            data, attempt = deepseek_request(base, key, model, messages, max_tokens=120, timeout=args.timeout, retries=2)
        except RuntimeError as exc:
            acct["failures"] += 1
            append_jsonl(run / "judge_deepseek.jsonl", {"text_id": g["text_id"], "error": str(exc)})
            write(acct_path, acct)
            continue
        acct["requests"] += 1
        acct["retries"] += attempt
        usage = data.get("usage", {})
        acct["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        acct["completion_tokens"] += int(usage.get("completion_tokens", 0))
        acct["cached_prompt_tokens"] += int(usage.get("prompt_cache_hit_tokens", 0))
        reported = data.get("model", "")
        acct["models_reported"][reported] = acct["models_reported"].get(reported, 0) + 1
        acct["system_fingerprint"] = data.get("system_fingerprint")
        content = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"parse_error": content[:200]}
        append_jsonl(run / "judge_deepseek.jsonl", {"text_id": g["text_id"], "model_reported": reported, "usage": usage, "rating": parsed})
        write(acct_path, acct)
    write(acct_path, acct | {"status": "COMPLETE"})
    print(json.dumps({k: v for k, v in acct.items() if k != "models_reported"} | {"models_reported": acct["models_reported"]}))


# ---------------------------------------------------------------- localize (teacher-forced patching)
def continuation_logprob(observer, ids, new_ids, patch=None, extra_hook=None):
    """Mean log p of the continuation tokens under the prompt (teacher forced)."""
    import torch

    full = ids + new_ids
    if extra_hook is not None:
        with extra_hook(), observer.hooks([len(ids) - 1], patch):
            output = observer._forward(full[:-1])
    else:
        with observer.hooks([len(ids) - 1], patch):
            output = observer._forward(full[:-1])
    logits = output.logits[0, len(ids) - 1: len(full) - 1].float()
    target = torch.tensor(new_ids, device=logits.device)
    lp = torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    return float(lp.mean().item())


def capture_states(observer, ids, positions):
    """Residual states at all block sites for the given positions."""
    captured, _ = observer.capture(ids, positions=positions)
    return {site: captured[site][: len(positions)] for site in captured if site.startswith("block_")}


def stage_localize(run, args):
    from grain_research.vector_mechanisms.trace import Patch

    manifest = json.loads((run / "manifest.json").read_text())
    roles = manifest["split_roles"]
    gens = {g["text_id"]: g for g in read_jsonl(run / "generations.jsonl")}
    rows = json.loads((run / "materials.json").read_text())
    seed = manifest["spec"]["seeds"][0]
    pairs = []
    for r in rows:
        if r["pole"] == 1 and r["seed"] == seed and r["situation"] in roles["discovery"] + roles["development"]:
            low_id = r["text_id"].replace(f"{r['axis']}_high", f"{r['axis']}_low")
            if r["text_id"] in gens and low_id in gens:
                pairs.append((gens[r["text_id"]], gens[low_id], "discovery" if r["situation"] in roles["discovery"] else "development"))
    if args.limit:
        pairs = pairs[: args.limit]
    observer = load_observer(max_tokens=1024, layer_count=28)
    layers = [s for s in observer.sites if s.startswith("block_")]
    coarse = layers[:: args.layer_step] if args.layer_step > 1 else layers
    rng = np.random.default_rng(20260914)
    done = {(r["pair"], r["site"], r["kind"], r["continuation"]) for r in read_jsonl(run / "localize_effects.jsonl")}
    start = time.monotonic()
    for high, low, role in pairs:
        pair_id = f"{high['axis']}|{high['family']}|{high['situation']}|s{seed}"
        ids_h, ids_l = high["prompt_ids"], low["prompt_ids"]
        span_h, span_l = high["persona_span"], low["persona_span"]
        if len(span_h) != len(span_l) or len(ids_h) != len(ids_l):
            append_jsonl(run / "localize_effects.jsonl", {"pair": pair_id, "kind": "skipped_unaligned"})
            continue
        n_new = min(len(high["new_ids"]), len(low["new_ids"]), args.max_continuation)
        conts = {"X_high": high["new_ids"][:n_new], "X_low": low["new_ids"][:n_new]}
        if n_new < 16:
            append_jsonl(run / "localize_effects.jsonl", {"pair": pair_id, "kind": "skipped_short"})
            continue
        base = {}
        for cname, cont in conts.items():
            base[cname] = {"high_prompt": continuation_logprob(observer, ids_h, cont), "low_prompt": continuation_logprob(observer, ids_l, cont)}
        append_jsonl(run / "localize_effects.jsonl", {"pair": pair_id, "role": role, "site": "none", "kind": "baseline", "continuation": "both", "values": base, "n_new": n_new, "span_tokens": len(span_h)})
        donor_states = capture_states(observer, ids_h, span_h)
        recipient_states = capture_states(observer, ids_l, span_l)
        suffix = list(range(span_l[-1] + 1, len(ids_l)))
        donor_suffix = capture_states(observer, ids_h, suffix) if args.suffix_patch else None
        for site in coarse:
            if time.monotonic() - start > args.max_seconds:
                write(run / "localize_status.json", {"status": "PARTIAL_BUDGET", "seconds": time.monotonic() - start})
                return
            for cname, cont in conts.items():
                ref_low, ref_high = base[cname]["low_prompt"], base[cname]["high_prompt"]
                kinds = {}
                if (pair_id, site, "persona_patch", cname) not in done:
                    kinds["persona_patch"] = Patch(site=site, position=span_l, donor=donor_states[site])
                if (pair_id, site, "random_equal_norm", cname) not in done:
                    delta = donor_states[site] - recipient_states[site]
                    rnd = rng.standard_normal(delta.shape)
                    rnd *= np.linalg.norm(delta, axis=1, keepdims=True) / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-9)
                    kinds["random_equal_norm"] = Patch(site=site, position=span_l, donor=recipient_states[site] + rnd)
                if site == coarse[len(coarse) // 2] and (pair_id, site, "self_patch", cname) not in done:
                    kinds["self_patch"] = Patch(site=site, position=span_l, donor=recipient_states[site])
                if args.suffix_patch and (pair_id, site, "suffix_patch", cname) not in done:
                    kinds["suffix_patch"] = Patch(site=site, position=suffix, donor=donor_suffix[site])
                for kind, patch in kinds.items():
                    lp = continuation_logprob(observer, ids_l, cont, patch=patch)
                    denom = ref_high - ref_low
                    append_jsonl(run / "localize_effects.jsonl", {"pair": pair_id, "role": role, "site": site, "kind": kind, "continuation": cname,
                                                                  "lp_patched": lp, "lp_low_prompt": ref_low, "lp_high_prompt": ref_high,
                                                                  "delta_lp": lp - ref_low, "recovery": (lp - ref_low) / denom if abs(denom) > 1e-9 else None,
                                                                  "forwards": observer.calls})
        print(f"pair {pair_id} done forwards={observer.calls} t={time.monotonic() - start:.0f}s", flush=True)
    write(run / "localize_status.json", {"status": "COMPLETE", "pairs": len(pairs), "forwards": observer.calls, "seconds": time.monotonic() - start, "sites": coarse, "suffix_patch": bool(args.suffix_patch)})


def stage_components(run, args):
    """MLP versus attention output contributions at selected layers (persona span)."""

    from grain_research.vector_mechanisms.trace import Patch

    sel = json.loads((run / "selected_sites.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    roles = manifest["split_roles"]
    gens = {g["text_id"]: g for g in read_jsonl(run / "generations.jsonl")}
    rows = json.loads((run / "materials.json").read_text())
    seed = manifest["spec"]["seeds"][0]
    observer = load_observer(max_tokens=1024, layer_count=28)
    decoder = observer.model.model
    for site in sel["sites"]:
        idx = int(site.split("_")[1])
        observer.sites[f"{site}_attn"] = decoder.layers[idx].self_attn
        observer.sites[f"{site}_mlp"] = decoder.layers[idx].mlp
    start = time.monotonic()
    pairs = [(gens[r["text_id"]], gens[r["text_id"].replace(f"{r['axis']}_high", f"{r['axis']}_low")]) for r in rows
             if r["pole"] == 1 and r["seed"] == seed and r["situation"] in roles["discovery"] + roles["development"]
             and r["text_id"] in gens and r["text_id"].replace(f"{r['axis']}_high", f"{r['axis']}_low") in gens]
    if args.limit:
        pairs = pairs[: args.limit]
    for high, low in pairs:
        pair_id = f"{high['axis']}|{high['family']}|{high['situation']}|s{seed}"
        ids_h, ids_l, span = high["prompt_ids"], low["prompt_ids"], low["persona_span"]
        if len(ids_h) != len(ids_l):
            continue
        n_new = min(len(high["new_ids"]), len(low["new_ids"]), args.max_continuation)
        conts = {"X_high": high["new_ids"][:n_new], "X_low": low["new_ids"][:n_new]}
        donor = capture_states_any(observer, ids_h, span)
        for cname, cont in conts.items():
            ref_low = continuation_logprob(observer, ids_l, cont)
            ref_high = continuation_logprob(observer, ids_h, cont)
            for site in sel["sites"]:
                for comp in ("attn", "mlp"):
                    name = f"{site}_{comp}"
                    lp = continuation_logprob(observer, ids_l, cont, patch=Patch(site=name, position=span, donor=donor[name]))
                    denom = ref_high - ref_low
                    append_jsonl(run / "component_effects.jsonl", {"pair": pair_id, "site": name, "continuation": cname, "delta_lp": lp - ref_low, "recovery": (lp - ref_low) / denom if abs(denom) > 1e-9 else None})
                # per-head at o_proj input
                idx = int(site.split("_")[1])
                attn = decoder.layers[idx].self_attn
                heads = observer.model.config.num_attention_heads
                dim = observer.model.config.hidden_size // heads
                donor_x = capture_o_proj_input(observer, attn, ids_h)
                for head in range(heads):
                    handle = attn.o_proj.register_forward_pre_hook(make_head_prehook(donor_x, span, head, dim))
                    try:
                        lp = continuation_logprob(observer, ids_l, cont)
                    finally:
                        handle.remove()
                    denom = ref_high - ref_low
                    append_jsonl(run / "component_effects.jsonl", {"pair": pair_id, "site": f"{site}_head{head:02d}", "continuation": cname, "delta_lp": lp - ref_low, "recovery": (lp - ref_low) / denom if abs(denom) > 1e-9 else None})
        print(f"components {pair_id} forwards={observer.calls} t={time.monotonic() - start:.0f}s", flush=True)
        if time.monotonic() - start > args.max_seconds:
            write(run / "components_status.json", {"status": "PARTIAL_BUDGET"})
            return
    write(run / "components_status.json", {"status": "COMPLETE", "forwards": observer.calls, "seconds": time.monotonic() - start})


def capture_o_proj_input(observer, attn, ids):
    """Donor per-head concatenated attention outputs (o_proj input) for one prompt."""
    box = {}

    def cap_pre(_m, inputs):
        box["x"] = inputs[0].detach().clone()

    handle = attn.o_proj.register_forward_pre_hook(cap_pre)
    try:
        observer._forward(ids)
    finally:
        handle.remove()
    return box["x"]


def make_head_prehook(donor_x, span, head, dim):
    """Replace one head's slice at the persona span with the donor's values (prefill only)."""

    def pre(_m, inputs):
        x = inputs[0].clone()
        sl = slice(head * dim, (head + 1) * dim)
        if x.shape[1] > 1:
            x[0, span, sl] = donor_x[0, span, sl]
        return (x,)

    return pre


def capture_states_any(observer, ids, positions):
    captured, _ = observer.capture(ids, positions=positions)
    return {site: captured[site][: len(positions)] for site in captured}


# ---------------------------------------------------------------- endpoint (GRAIN on regenerated continuations)
def stage_endpoint(run, args):
    import torch

    manifest = json.loads((run / "manifest.json").read_text())
    roles = manifest["split_roles"]
    sel = json.loads((run / "selected_sites.json").read_text())
    site = sel["endpoint_site"]
    gens = {g["text_id"]: g for g in read_jsonl(run / "generations.jsonl")}
    rows = json.loads((run / "materials.json").read_text())
    gen = manifest["generation"]
    observer = load_observer(max_tokens=768, layer_count=28)
    decoder = observer.model.model
    idx = int(site.split("_")[1])
    block = decoder.layers[idx]
    eos = observer.tokenizer.eos_token_id
    device = next(observer.model.parameters()).device
    rng = np.random.default_rng(20260915)
    done = {(r["pair"], r["arm"], r["seed"]) for r in read_jsonl(run / "endpoint_generations.jsonl")}
    start = time.monotonic()
    base_seed = manifest["spec"]["seeds"][0]
    pairs = [r for r in rows if r["pole"] == 1 and r["seed"] == base_seed and r["situation"] in roles["evaluation"] + roles["development"]]
    if args.limit:
        pairs = pairs[: args.limit]

    def generate(ids, seed, donor=None, span=None):
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] > 1 and donor is not None:
                hidden = hidden.clone()
                hidden[0, span] = torch.as_tensor(donor, device=hidden.device, dtype=hidden.dtype)
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            return None

        handle = block.register_forward_hook(hook)
        try:
            torch.manual_seed(seed)
            tokens = torch.tensor([ids], device=device)
            with torch.inference_mode():
                out = observer.model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens), do_sample=True, temperature=gen["temperature"], top_p=gen["top_p"], max_new_tokens=args.new_tokens, pad_token_id=eos, eos_token_id=eos, use_cache=True)
        finally:
            handle.remove()
        new = out[0, len(ids):].tolist()
        new = new[: new.index(eos)] if eos in new else new
        return observer.tokenizer.decode(new, skip_special_tokens=True), len(new)

    for r in pairs:
        high = gens[r["text_id"]]
        low = gens[r["text_id"].replace(f"{r['axis']}_high", f"{r['axis']}_low")]
        pair_id = f"{r['axis']}|{r['family']}|{r['situation']}"
        ids_h, ids_l, span = high["prompt_ids"], low["prompt_ids"], low["persona_span"]
        if len(ids_h) != len(ids_l):
            continue
        donor = capture_states(observer, ids_h, span)[site]
        recipient = capture_states(observer, ids_l, span)[site]
        delta = donor - recipient
        rnd = rng.standard_normal(delta.shape)
        rnd *= np.linalg.norm(delta, axis=1, keepdims=True) / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-9)
        arms = {"control_low": (ids_l, None), "intervention_high_into_low": (ids_l, donor), "random_equal_norm": (ids_l, recipient + rnd), "reference_high_prompt": (ids_h, None)}
        for seed in args.seeds:
            for arm, (ids, dn) in arms.items():
                if (pair_id, arm, seed) in done:
                    continue
                if time.monotonic() - start > args.max_seconds:
                    write(run / "endpoint_status.json", {"status": "PARTIAL_BUDGET"})
                    return
                text, n = generate(ids, seed, dn, span)
                append_jsonl(run / "endpoint_generations.jsonl", {"pair": pair_id, "axis": r["axis"], "family": r["family"], "situation": r["situation"], "arm": arm, "seed": seed, "site": site, "text": text, "n_new": n, "audit": audit_text(text, r["situation"], "eos" if n < args.new_tokens else "length")})
        print(f"endpoint {pair_id} t={time.monotonic() - start:.0f}s", flush=True)
    write(run / "endpoint_status.json", {"status": "COMPLETE", "site": site, "seconds": time.monotonic() - start, "new_tokens": args.new_tokens, "seeds": args.seeds})


# ---------------------------------------------------------------- read side (K/V at persona positions)
def capture_kv(observer, decoder, ids, span):
    """Pre-RoPE key/value projections at the persona span for every layer (one forward)."""
    box = {}
    handles = []
    for i, layer in enumerate(decoder.layers):
        for name in ("k_proj", "v_proj"):
            handles.append(getattr(layer.self_attn, name).register_forward_hook(make_kv_capture(box, (i, name), span)))
    try:
        observer._forward(ids)
    finally:
        for h in handles:
            h.remove()
    return box


def make_kv_capture(box, key, span):
    def cap(_m, _i, out):
        box[key] = out[0, span].detach().clone()

    return cap


def make_kv_hook(donor_rows, span, kv_head=None, dim=None):
    """Replace the persona-span rows of a k_proj/v_proj output with the donor's (optionally one KV head)."""

    def hook(_m, _i, out):
        out = out.clone()
        if kv_head is None:
            out[0, span] = donor_rows.to(out.dtype)
        else:
            sl = slice(kv_head * dim, (kv_head + 1) * dim)
            out[0, span, sl] = donor_rows[:, sl].to(out.dtype)
        return out

    return hook


def kv_patch_context(decoder, donor, span, layers, kv_head=None, dim=None):
    """Context manager factory: donor K/V rows at the span for the listed layers."""
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        handles = []
        try:
            for i in layers:
                for name in ("k_proj", "v_proj"):
                    handles.append(getattr(decoder.layers[i].self_attn, name).register_forward_hook(make_kv_hook(donor[(i, name)], span, kv_head, dim)))
            yield
        finally:
            for h in handles:
                h.remove()

    return ctx


def make_row_capture(box, key):
    """Forward pre-hook: keep a module's input rows (o_proj input = concatenated head outputs)."""

    def pre(_m, inputs):
        box[key] = inputs[0].detach().clone()

    return pre


def make_row_replace(donor_x, positions, head=None, dim=None):
    """Forward pre-hook: replace o_proj input rows at `positions` with the donor's (optionally one query head)."""

    def pre(_m, inputs):
        x = inputs[0].clone()
        if head is None:
            x[0, positions] = donor_x[0, positions].to(x.dtype)
        else:
            sl = slice(head * dim, (head + 1) * dim)
            x[0, positions, sl] = donor_x[0, positions, sl].to(x.dtype)
        return (x,)

    return pre


def downstream_read(observer, decoder, ids, cont, span, donor_kv, layers, kv_head=None, kv_dim=None, query_head=None, q_dim=None):
    """Direct effect of downstream reads of the persona span at `layers` (two forwards).

    Forward 1: low prompt with donor K/V at the span in `layers`; keep the attention outputs
    (o_proj inputs) of those layers.  Forward 2: clean low prompt, but the attention outputs of
    those layers at NON-span positions are replaced by forward-1 rows.  The span's own residual
    stream therefore never changes: only what downstream positions read from the span does.
    """
    from contextlib import contextmanager

    n = len(ids) + len(cont) - 1
    downstream = [p for p in range(n) if p not in set(span)]
    box = {}

    @contextmanager
    def first():
        handles = []
        try:
            for i in layers:
                attn = decoder.layers[i].self_attn
                for name in ("k_proj", "v_proj"):
                    handles.append(getattr(attn, name).register_forward_hook(make_kv_hook(donor_kv[(i, name)], span, kv_head, kv_dim)))
                handles.append(attn.o_proj.register_forward_pre_hook(make_row_capture(box, i)))
            yield
        finally:
            for h in handles:
                h.remove()

    continuation_logprob(observer, ids, cont, extra_hook=first)

    @contextmanager
    def second():
        handles = []
        try:
            for i in layers:
                handles.append(decoder.layers[i].self_attn.o_proj.register_forward_pre_hook(make_row_replace(box[i], downstream, query_head, q_dim)))
            yield
        finally:
            for h in handles:
                h.remove()

    return continuation_logprob(observer, ids, cont, extra_hook=second)


def _readside_pairs(run, args):
    manifest = json.loads((run / "manifest.json").read_text())
    roles = manifest["split_roles"]
    gens = {g["text_id"]: g for g in read_jsonl(run / "generations.jsonl")}
    rows = json.loads((run / "materials.json").read_text())
    seed = manifest["spec"]["seeds"][0]
    out = []
    for r in rows:
        if r["pole"] != 1 or r["seed"] != seed or r["situation"] not in roles["discovery"] + roles["development"]:
            continue
        low_id = r["text_id"].replace(f"{r['axis']}_high", f"{r['axis']}_low")
        if r["text_id"] not in gens or low_id not in gens:
            continue
        high, low = gens[r["text_id"]], gens[low_id]
        if len(high["prompt_ids"]) != len(low["prompt_ids"]):
            continue
        n_new = min(len(high["new_ids"]), len(low["new_ids"]), args.max_continuation)
        out.append({"pair": f"{r['axis']}|{r['family']}|{r['situation']}|s{seed}", "role": "discovery" if r["situation"] in roles["discovery"] else "development",
                    "ids_h": high["prompt_ids"], "ids_l": low["prompt_ids"], "span": low["persona_span"], "conts": {"X_high": high["new_ids"][:n_new], "X_low": low["new_ids"][:n_new]}})
    return out[: args.limit] if args.limit else out


def stage_readside(run, args):
    """Where downstream positions READ the persona span: path-patched direct effect per layer.

    v1 (`readside_effects_v0_confounded.jsonl`, if present) replaced the span K/V in place, which also
    changed the span's own residual stream through within-span attention; this version keeps the
    span at the low pole and only hands downstream positions the donor's read (see downstream_read).
    """
    observer = load_observer(max_tokens=1024, layer_count=28)
    decoder = observer.model.model
    n_layers = len(decoder.layers)
    done = {(r["pair"], r["site"], r["continuation"]) for r in read_jsonl(run / "readside_effects.jsonl")}
    start = time.monotonic()
    for p in _readside_pairs(run, args):
        donor = capture_kv(observer, decoder, p["ids_h"], p["span"])
        sites = [("read_all_layers", list(range(n_layers)))] + [(f"read_layer_{i:02d}", [i]) for i in range(n_layers)]
        for cname, cont in p["conts"].items():
            ref_low = continuation_logprob(observer, p["ids_l"], cont)
            ref_high = continuation_logprob(observer, p["ids_h"], cont)
            denom = ref_high - ref_low
            for site, layers in sites:
                if (p["pair"], site, cname) in done:
                    continue
                lp = downstream_read(observer, decoder, p["ids_l"], cont, p["span"], donor, layers)
                append_jsonl(run / "readside_effects.jsonl", {"pair": p["pair"], "role": p["role"], "site": site, "continuation": cname, "lp_patched": lp, "lp_low_prompt": ref_low, "lp_high_prompt": ref_high,
                                                             "delta_lp": lp - ref_low, "recovery": (lp - ref_low) / denom if abs(denom) > 1e-9 else None, "span_tokens": len(p["span"]), "n_new": len(cont)})
        print(f"readside {p['pair']} forwards={observer.calls} t={time.monotonic() - start:.0f}s", flush=True)
        if time.monotonic() - start > args.max_seconds:
            write(run / "readside_status.json", {"status": "PARTIAL_BUDGET", "forwards": observer.calls})
            return
    write(run / "readside_status.json", {"status": "COMPLETE", "forwards": observer.calls, "seconds": time.monotonic() - start, "design": "path-patched downstream reads, span residual kept at low pole"})


def stage_readside_heads(run, args):
    """Per-KV-head (sender side) and per-query-head (receiver side) direct reads at chosen layers."""
    observer = load_observer(max_tokens=1024, layer_count=28)
    decoder = observer.model.model
    cfg = observer.model.config
    n_kv, n_q = cfg.num_key_value_heads, cfg.num_attention_heads
    q_dim = cfg.hidden_size // n_q
    layers = [int(x) for x in args.kv_head_layers.split(",") if x.strip()]
    done = {(r["pair"], r["site"], r["continuation"]) for r in read_jsonl(run / "readside_head_effects.jsonl")}
    start = time.monotonic()
    for p in _readside_pairs(run, args):
        donor = capture_kv(observer, decoder, p["ids_h"], p["span"])
        for cname, cont in p["conts"].items():
            ref_low = continuation_logprob(observer, p["ids_l"], cont)
            ref_high = continuation_logprob(observer, p["ids_h"], cont)
            denom = ref_high - ref_low
            for i in layers:
                jobs = [(f"read_layer_{i:02d}_kvhead{h}", {"kv_head": h, "kv_dim": q_dim}) for h in range(n_kv)]
                jobs += [(f"read_layer_{i:02d}_qhead{h:02d}", {"query_head": h, "q_dim": q_dim}) for h in range(n_q)]
                for site, kw in jobs:
                    if (p["pair"], site, cname) in done:
                        continue
                    lp = downstream_read(observer, decoder, p["ids_l"], cont, p["span"], donor, [i], **kw)
                    append_jsonl(run / "readside_head_effects.jsonl", {"pair": p["pair"], "role": p["role"], "site": site, "layer": i, "continuation": cname, "lp_patched": lp, "lp_low_prompt": ref_low, "lp_high_prompt": ref_high,
                                                                      "delta_lp": lp - ref_low, "recovery": (lp - ref_low) / denom if abs(denom) > 1e-9 else None})
        print(f"readside_heads {p['pair']} forwards={observer.calls} t={time.monotonic() - start:.0f}s", flush=True)
        if time.monotonic() - start > args.max_seconds:
            write(run / "readside_heads_status.json", {"status": "PARTIAL_BUDGET", "forwards": observer.calls})
            return
    write(run / "readside_heads_status.json", {"status": "COMPLETE", "forwards": observer.calls, "seconds": time.monotonic() - start, "layers": layers, "kv_heads": n_kv, "query_heads": n_q})


def stage_norms(run, args):
    """Geometry of the high-low persona-span difference at every block: relative norm and cosine."""
    manifest = json.loads((run / "manifest.json").read_text())
    roles = manifest["split_roles"]
    gens = {g["text_id"]: g for g in read_jsonl(run / "generations.jsonl")}
    rows = json.loads((run / "materials.json").read_text())
    seed = manifest["spec"]["seeds"][0]
    observer = load_observer(max_tokens=1024, layer_count=28)
    out = []
    for r in rows:
        if r["pole"] != 1 or r["seed"] != seed or r["situation"] not in roles["discovery"] + roles["development"]:
            continue
        low_id = r["text_id"].replace(f"{r['axis']}_high", f"{r['axis']}_low")
        if r["text_id"] not in gens or low_id not in gens:
            continue
        high, low = gens[r["text_id"]], gens[low_id]
        if len(high["prompt_ids"]) != len(low["prompt_ids"]):
            continue
        span = low["persona_span"]
        donor = capture_states_any(observer, high["prompt_ids"], span)
        recip = capture_states_any(observer, low["prompt_ids"], span)
        for site in sorted(donor):
            d, rc = np.asarray(donor[site], dtype=np.float64), np.asarray(recip[site], dtype=np.float64)
            delta = d - rc
            rel = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(rc, axis=1), 1e-9)
            cos = np.sum(d * rc, axis=1) / np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(rc, axis=1), 1e-9)
            out.append({"pair": f"{r['axis']}|{r['family']}|{r['situation']}|s{seed}", "role": "discovery" if r["situation"] in roles["discovery"] else "development", "site": site,
                        "rel_delta_norm_mean": float(rel.mean()), "rel_delta_norm_max": float(rel.max()), "cos_high_low_mean": float(cos.mean()), "cos_high_low_min": float(cos.min()), "span_tokens": len(span)})
    write(run / "span_norms.json", {"rows": out, "forwards": observer.calls, "note": "per persona-span position: ||high-low|| / ||low|| and cos(high, low) at each block output"})
    print(json.dumps({"pairs": len({o["pair"] for o in out}), "forwards": observer.calls}))


def stage_endpoint_read(run, args):
    """Frozen GRAIN coordinates and the local text-only readout on regenerated continuations."""
    gens = read_jsonl(run / "endpoint_generations.jsonl")
    texts = {f"{g['pair']}|{g['arm']}|s{g['seed']}": g["text"] for g in gens}
    write(run / "endpoint_grain.json", grain_coordinates(texts))
    observer = load_observer(max_tokens=1024, layer_count=2)
    start = time.monotonic()
    local = local_query_readout(observer, texts, ("extraversion", "neuroticism"))
    write(run / "endpoint_local_query.json", {"raw": local, "forwards": observer.calls, "seconds": time.monotonic() - start})
    print(json.dumps({"endpoint_texts": len(texts), "forwards": observer.calls}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "read", "judge", "localize", "components", "norms", "readside", "readside_heads", "endpoint", "endpoint_read"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--judge-model", default="deepseek-flash")
    parser.add_argument("--max-requests", type=int, default=400)
    parser.add_argument("--max-total-tokens", type=int, default=400000)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--layer-step", type=int, default=2)
    parser.add_argument("--max-continuation", type=int, default=96)
    parser.add_argument("--suffix-patch", action="store_true")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 23, 37])
    parser.add_argument("--kv-head-layers", default="14,15,16,17", help="layers for per-head read-side patching (readside_heads)")
    args = parser.parse_args()
    run = args.run.resolve()
    globals()[f"stage_{args.stage}"](run, args)


if __name__ == "__main__":
    main()

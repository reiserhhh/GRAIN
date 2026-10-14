#!/usr/bin/env python3
"""P05 native-action packet runner (docs/GRAIN_P05_NATIVE_ACTION_PACKET_V1.md). Stages: prepare, replay, jacobian, fit, apply, linearize, analyze."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.native_action import (
    Accounting,
    apply_rank1_dual,
    euclid_objective,
    fit_euclid,
    fit_rank1_dual,
    metric_scale,
    metric_spectrum,
    rank1_objective,
    source_scale,
)

PROTOCOL = ROOT / "docs/GRAIN_P05_NATIVE_ACTION_PACKET_V1.md"
PROTOCOL_SHA = "f137c7ff7d3dc4ba55527558037bbdc970d6c5552f7114650527bc7323bc194e"
IDS = ROOT / "docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
GEN = ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1/expanded/generations.jsonl"
RUNTIME = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
OUT = ROOT / "results/grain_gptpro_foundation_v3/P05/native_action_r1"
QUERIES = ("extraversion", "neuroticism")
CAP, ALPHA, H_FD, TOL_REPLAY, TOL_FD = 960, 1.0, 0.1, 1e-5, 0.10
CELLS = ("E4", "J4", "E24", "J24")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    for rel, sha in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if digest(ROOT / rel) != sha:
            raise ValueError(f"Binding mismatch: {rel}")


def load_panel():
    arr = np.load(OP / "action_arrays.npz", allow_pickle=False)
    order = json.loads((OP / "pair_order.json").read_text())
    pairs = pd.read_csv(OP / "pairs.csv").set_index("pair").loc[order].reset_index()
    ids = json.loads(IDS.read_text())
    src = arr["source"]
    if pairs.pair[src].tolist() != ids["source_pairs"] or pairs.pair[~src].tolist() != ids["application_pairs"]:
        raise ValueError("Pair ID mismatch")
    return arr, pairs, ids


def state(rid, q):
    key = hashlib.sha256(f"{rid}|{q}".encode()).hexdigest()
    with np.load(OP / "states" / f"{key}.npz") as z:
        return {k: z[k] for k in z.files}


def segments(pairs):
    shapes = [state(pairs.low_id[0], q)["head"].shape for q in QUERIES]
    off = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    return shapes, off


def texts():
    return {json.loads(x)["text_id"]: json.loads(x)["text"] for x in GEN.read_text().splitlines()}


def stage_prepare(args):
    import joblib

    verify()
    arr, pairs, _ids = load_panel()
    src = arr["source"]
    shapes, off = segments(pairs)
    tx = texts()
    rt = joblib.load(RUNTIME)
    de24 = np.asarray(rt.transform([tx[t] for t in pairs.high_id]) - rt.transform([tx[t] for t in pairs.low_id]), float)
    out = {}
    for name, de in (("4", arr["dx"].astype(float)), ("24", de24)):
        scale, keep = source_scale(de[src])
        z = (de / scale)[:, keep]
        out[f"z{name}"] = z
        out[f"keep{name}"] = keep
    Y = arr["dh"].astype(float)
    report = {"shapes": [list(s) for s in shapes], "cells": {}}
    for name in ("4", "24"):
        Z = out[f"z{name}"]
        comp = np.zeros_like(Y)
        for qi, q in enumerate(QUERIES):
            sl = slice(off[qi], off[qi + 1])
            W = fit_euclid(Z[src], Y[src, sl], ALPHA)
            comp[:, sl] = Z @ W
            report["cells"][f"E{name}|{q}"] = {"objective": euclid_objective(W, Z[src], Y[src, sl], ALPHA),
                                               "rank": int(np.linalg.matrix_rank(W)),
                                               "source_fit_rel_err": float(np.linalg.norm(Y[src, sl] - Z[src] @ W) / np.linalg.norm(Y[src, sl]))}
        out[f"E{name}"] = comp
    out["zmax_source_app"] = np.array([[np.abs(out["z4"][src]).max(), np.abs(out["z4"][~src]).max()],
                                       [np.abs(out["z24"][src]).max(), np.abs(out["z24"][~src]).max()]])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "private_prepared.npz", **out)
    report["zmax_source_app"] = out["zmax_source_app"].tolist()
    (OUT / "prepare.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def observer_and_alphabets():
    from grain_research.vector_mechanisms.direct_traits import candidates
    from scripts.run_grain_persona_generation_trace import load_observer

    obs = load_observer(max_tokens=1024, layer_count=2)
    return obs, {q: candidates(q, obs.tokenizer) for q in QUERIES}


def check_suffix(obs, pairs, rows):
    from scripts.run_grain_conditional_operator_native import context

    tx = texts()
    for rid in rows:
        for q in QUERIES:
            ids, pos = context(obs, tx[rid], q)
            if [ids[i] for i in pos] != state(rid, q)["suffix"].tolist():
                raise ValueError(f"Suffix binding mismatch {rid} {q}")


def grad_score(obs, ids, pos, alphabet):
    import torch

    cfg = obs.model.config
    dim = cfg.hidden_size // cfg.num_attention_heads
    dev = next(obs.model.parameters()).device
    delta = torch.zeros(len(pos), dim, device=dev, dtype=torch.float32, requires_grad=True)
    idx = torch.tensor(pos, device=dev)

    def hook(_m, xs):
        x = xs[0].clone()
        x[0, idx, :dim] = x[0, idx, :dim] + delta
        return (x,)

    handle = obs.model.model.layers[12].self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        tokens = torch.tensor([ids], dtype=torch.long, device=dev)
        mask = torch.ones_like(tokens)
        with torch.enable_grad():
            out = obs.model(input_ids=tokens, attention_mask=mask, use_cache=False, return_dict=True)
            local = out.logits[0, len(ids) - 1][torch.tensor(alphabet["token_ids"], device=dev)]
            q = torch.softmax(local, 0)
            score = (q * torch.tensor(alphabet["values"], device=dev, dtype=q.dtype)).sum()
            score.backward()
    finally:
        handle.remove()
    return float(score.detach().cpu()), delta.grad.detach().cpu().numpy().astype(np.float64)


def call(acc, item, step, passes, fn):
    done = acc.done()
    if item in done:
        return done[item]
    acc.start(item, step, passes)
    try:
        payload = fn()
    except Exception as exc:
        acc.fail(item, repr(exc))
        raise
    acc.finish(item, **payload)
    return acc.done()[item]


def stage_replay(args):
    from scripts.run_grain_conditional_operator_native import context, intervene
    from scripts.run_grain_persona_reverse_trace import read_score

    verify()
    arr, pairs, ids = load_panel()
    shapes, off = segments(pairs)
    obs, alpha = observer_and_alphabets()
    tx = texts()
    check_suffix(obs, pairs, pairs.low_id.tolist())
    acc = Accounting(OUT / "accounting.jsonl", CAP)
    eff = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    donor = eff[eff.kind == "donor"].set_index(["pair", "query"])["effect"]
    order = pairs.pair.tolist()
    fails = []
    for p in ids["replay_pairs"]:
        i = order.index(p)
        low = pairs.low_id[i]
        for qi, q in enumerate(QUERIES):
            c_ids, pos = context(obs, tx[low], q)
            cached = float(state(low, q)["score"])
            r = call(acc, f"replay_low|{p}|{q}", "replay", 1, lambda c_ids=c_ids, q=q: {"score": read_score(obs, c_ids, alpha[q])["score"]})
            if abs(r["score"] - cached) > TOL_REPLAY:
                fails.append(("low", p, q, r["score"], cached))
            d = arr["dh"][i, off[qi]:off[qi + 1]].reshape(shapes[qi])
            r2 = call(acc, f"replay_donor|{p}|{q}", "replay", 1, lambda c_ids=c_ids, pos=pos, q=q, d=d: {"score": intervene(obs, c_ids, pos, alpha[q], d)["score"]})
            if abs((r2["score"] - cached) - donor[(p, q)]) > TOL_REPLAY:
                fails.append(("donor", p, q, r2["score"] - cached, float(donor[(p, q)])))
    status = {"replay_fails": fails, "passes_so_far": acc.passes_attempted(), "suffix_binding": "OK"}
    (OUT / "replay.json").write_text(json.dumps(status, indent=1) + "\n")
    print(json.dumps(status, indent=1))
    if fails:
        raise SystemExit("BLOCKED: replay tolerance failed")


def stage_jacobian(args):
    from scripts.run_grain_conditional_operator_native import context, intervene

    verify()
    _arr, pairs, ids = load_panel()
    _shapes, _off = segments(pairs)
    obs, alpha = observer_and_alphabets()
    tx = texts()
    acc = Accounting(OUT / "accounting.jsonl", CAP)
    order = pairs.pair.tolist()
    jdir = OUT / "private_jacobians"
    jdir.mkdir(exist_ok=True)
    worst = 0.0
    for p in ids["source_pairs"]:
        i = order.index(p)
        low = pairs.low_id[i]
        for q in QUERIES:
            target = jdir / f"{i:03d}_{q}.npy"

            def fn(low=low, q=q, target=target):
                c_ids, pos = context(obs, tx[low], q)
                s, g = grad_score(obs, c_ids, pos, alpha[q])
                np.save(target, g)
                return {"score": s, "cached_low": float(state(low, q)["score"]), "grad_norm": float(np.linalg.norm(g)), "sha256": digest(target)}
            r = call(acc, f"jacobian|{p}|{q}", "jacobian", 2, fn)
            worst = max(worst, abs(r["score"] - r["cached_low"]))
    if worst > TOL_REPLAY:
        raise SystemExit(f"BLOCKED: grad-pass score replay {worst}")
    fd = []
    for p in ids["fd_check_source_pairs"]:
        i = order.index(p)
        low = pairs.low_id[i]
        q = "extraversion"
        g = np.load(jdir / f"{i:03d}_{q}.npy")
        unit = g / np.linalg.norm(g)
        c_ids, pos = context(obs, tx[low], q)
        sp = call(acc, f"fd_plus|{p}|{q}", "fd", 1, lambda c_ids=c_ids, pos=pos, q=q, unit=unit: {"score": intervene(obs, c_ids, pos, alpha[q], H_FD * unit)["score"]})
        sm = call(acc, f"fd_minus|{p}|{q}", "fd", 1, lambda c_ids=c_ids, pos=pos, q=q, unit=unit: {"score": intervene(obs, c_ids, pos, alpha[q], -H_FD * unit)["score"]})
        cd = (sp["score"] - sm["score"]) / (2 * H_FD)
        fd.append({"pair": p, "central_diff": cd, "grad_norm": float(np.linalg.norm(g)), "rel_err": abs(cd - np.linalg.norm(g)) / np.linalg.norm(g)})
    status = {"grad_pass_score_replay_max_abs": worst, "fd_check": fd, "passes_so_far": acc.passes_attempted()}
    (OUT / "jacobian.json").write_text(json.dumps(status, indent=1) + "\n")
    print(json.dumps(status, indent=1))
    if any(x["rel_err"] > TOL_FD for x in fd):
        raise SystemExit("BLOCKED: finite-difference check failed")


def stage_fit(args):
    verify()
    arr, pairs, _ids = load_panel()
    _shapes, off = segments(pairs)
    src = arr["source"]
    prep = dict(np.load(OUT / "private_prepared.npz"))
    Y = arr["dh"].astype(float)
    src_idx = np.flatnonzero(src)
    rep = {"cells": {}}
    for name in ("4", "24"):
        Z = prep[f"z{name}"]
        comp = np.zeros_like(Y)
        for qi, q in enumerate(QUERIES):
            sl = slice(off[qi], off[qi + 1])
            J = np.stack([np.load(OUT / "private_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in src_idx])
            s = metric_scale(J)
            U = np.sqrt(s) * J
            T = (U * Y[src, sl]).sum(1)
            c, K = fit_rank1_dual(Z[src], U, T, ALPHA)
            comp[:, sl] = apply_rank1_dual(c, Z[src], U, Z)
            w, rank = metric_spectrum(U)
            rep["cells"][f"J{name}|{q}"] = {"metric_scale_s": s, "objective": rank1_objective(c, K, T, ALPHA),
                                            "metric_rank": rank, "nullspace_dim": int(J.shape[1] - rank),
                                            "metric_eig_top5": w[:5].tolist(), "metric_eig_min_nonzero": float(w[rank - 1]),
                                            "source_linear_action_fit_rel_err": float(np.linalg.norm(T - K @ c) / np.linalg.norm(T))}
        prep[f"J{name}"] = comp
    np.savez(OUT / "private_prepared.npz", **prep)
    donor_norm = {q: np.linalg.norm(Y[~src, off[qi]:off[qi + 1]], axis=1) for qi, q in enumerate(QUERIES)}
    for cell in CELLS:
        for qi, q in enumerate(QUERIES):
            a = prep[cell][~src, off[qi]:off[qi + 1]]
            rep.setdefault("norm_ratio_median", {})[f"{cell}|{q}"] = float(np.median(np.linalg.norm(a, axis=1) / donor_norm[q]))
    (OUT / "fit.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def stage_apply(args):
    from grain_research.vector_mechanisms.reverse_trace import equal_norm_random
    from scripts.run_grain_conditional_operator_native import context, intervene

    verify()
    arr, pairs, ids = load_panel()
    shapes, off = segments(pairs)
    arr["source"]
    prep = dict(np.load(OUT / "private_prepared.npz"))
    obs, alpha = observer_and_alphabets()
    tx = texts()
    acc = Accounting(OUT / "accounting.jsonl", CAP)
    order = pairs.pair.tolist()
    rows_path = OUT / "effects.jsonl"
    for p in ids["application_pairs"]:
        i = order.index(p)
        low = pairs.low_id[i]
        for qi, q in enumerate(QUERIES):
            c_ids, pos = context(obs, tx[low], q)
            cached = float(state(low, q)["score"])
            deltas = {cell: prep[cell][i, off[qi]:off[qi + 1]].reshape(shapes[qi]) for cell in CELLS}
            deltas["RAND_J24"] = equal_norm_random(np.zeros(shapes[qi]), deltas["J24"], f"jw24|{p}|{q}")
            for kind, d in deltas.items():
                item = f"apply|{p}|{q}|{kind}"
                if item in acc.done():
                    continue

                def fn(d=d, kind=kind, q=q, cached=cached, p=p, c_ids=c_ids, pos=pos):
                    r = intervene(obs, c_ids, pos, alpha[q], d)
                    rec = {"pair": p, "query": q, "kind": kind, "score": r["score"], "effect": r["score"] - cached,
                           "mass": r["mass"], "action_norm": float(np.linalg.norm(d))}
                    with rows_path.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    return rec
                call(acc, item, "apply", 1, fn)
        print("applied", p, acc.passes_attempted(), flush=True)


def stage_linearize(args):
    from scripts.run_grain_conditional_operator_native import context, intervene

    verify()
    _arr, pairs, ids = load_panel()
    shapes, off = segments(pairs)
    prep = dict(np.load(OUT / "private_prepared.npz"))
    obs, alpha = observer_and_alphabets()
    tx = texts()
    acc = Accounting(OUT / "accounting.jsonl", CAP)
    order = pairs.pair.tolist()
    rows_path = OUT / "linearization.jsonl"
    for p in ids["source_pairs"]:
        i = order.index(p)
        low = pairs.low_id[i]
        for qi, q in enumerate(QUERIES):
            d = prep["J24"][i, off[qi]:off[qi + 1]]
            J = np.load(OUT / "private_jacobians" / f"{i:03d}_{q}.npy").ravel()
            c_ids, pos = context(obs, tx[low], q)
            cached = float(state(low, q)["score"])

            def fn(d=d, J=J, q=q, cached=cached, p=p, c_ids=c_ids, pos=pos, shape=shapes[qi]):
                r = intervene(obs, c_ids, pos, alpha[q], d.reshape(shape))
                rec = {"pair": p, "query": q, "predicted_linear": float(J @ d), "actual": r["score"] - cached,
                       "action_norm": float(np.linalg.norm(d))}
                with rows_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                return rec
            call(acc, f"linearize|{p}|{q}", "linearize", 1, fn)
    print("linearize done", acc.passes_attempted())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "replay", "jacobian", "fit", "apply", "linearize"])
    args = ap.parse_args()
    {"prepare": stage_prepare, "replay": stage_replay, "jacobian": stage_jacobian, "fit": stage_fit,
     "apply": stage_apply, "linearize": stage_linearize}[args.stage](args)


if __name__ == "__main__":
    main()

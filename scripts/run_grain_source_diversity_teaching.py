#!/usr/bin/env python3
"""NEXT88: exact disjoint material preparation, bounded capture and reader fitting."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms import source_diversity_teaching as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_compiled_compatible_reader as previous
from scripts import run_grain_reader_new_author_transport as material

old, ep, p85 = previous.old, previous.ep, previous.p85
OUT = old.BASE / "logic_source_diversity_teaching_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_DIVERSITY_TEACHING_PLAN_V1.md"
CAP, NATIVE_CAP = 120, 900
CORE, TABLES = previous.CORE, previous.TABLES
COUNTS = [528, 1584, 6336, 286, 22, 2112, 10, 2, 20, 32]


def limit(start, used):
    """Bound numerical construction/application separately from native capture."""
    if used + time.process_time() - start > CAP:
        raise TimeoutError("NEXT88 numerical CPU budget")


def prepare(out):
    """Freeze selected material and same GRAIN/tokenizer protocol without labels."""
    import joblib
    from transformers import AutoTokenizer

    old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT87 required")
    start = time.monotonic()
    path = material.SOURCE / "private_material.json"
    if old.sha(path) != old.read(material.SOURCE / "manifest.json")["material_sha256"]:
        raise ValueError("Frozen original material changed")
    source_path = material.RESIDUAL / "private_membership.json"
    eval_path = material.INVENTORY / "private_selection.json"
    sources = {r["author_id"] for r in old.read(source_path)["authors"]}
    recipients = {r["author_id"] for r in old.read(eval_path)["authors"]}
    selected, counts = fn.select(old.read(path), sources, recipients)
    tokenizer = AutoTokenizer.from_pretrained(
        str(material.MODEL), local_files_only=True, trust_remote_code=False
    )
    rows, metadata, design, allocations = [], [], [], {}
    for i, item in enumerate(selected):
        texts = item["texts"]
        if len(texts) != 12:
            raise ValueError("Original twelve-comment view0 required")
        encoded, spans, queries = {}, {}, {}
        for q in old.QUERIES:
            _, spans[q], encoded[q] = material.token_view(tokenizer, texts, q)
            queries[q] = {
                "ids": list(encoded[q]["input_ids"]),
                "alphabet": material.native_alphabet(q, tokenizer),
            }
        prefix = material.common_prefix_length([queries[q]["ids"] for q in old.QUERIES])
        body_end = (
            max(
                j
                for j, (_, hi) in enumerate(encoded[old.QUERIES[0]]["offset_mapping"])
                if hi <= spans[old.QUERIES[0]][-1][1]
            )
            + 1
        )
        if prefix < body_end or spans[old.QUERIES[0]] != spans[old.QUERIES[1]]:
            raise ValueError("Query changes the common material token prefix")
        slice_spans, eligible = [], 0
        for j, text in enumerate(texts):
            parts = material.lexical_slices(text)
            if not parts:
                continue
            eligible += 1
            lo = spans[old.QUERIES[0]][j][0]
            for s, part in enumerate(parts):
                key = f"a{i:02d}_c{j:02d}_s{s}"
                rows.append({"author": key, "body": part})
                metadata.append({"user_id": key, "index": i, "comment": j, "slice": s})
                slice_spans.append((lo, lo + len(part)))
                lo += len(part)
        if eligible < 2:
            raise ValueError("Insufficient source slices; no replacement")
        allocations[i] = (encoded[old.QUERIES[0]]["offset_mapping"], slice_spans)
        design.append(
            {
                "index": i,
                "author_id": item["author_id"],
                "source_group": item["source_group"],
                "view": 0,
                "comments": 12,
                "eligible_comments": eligible,
                "text_list_sha256": hashlib.sha256(
                    json.dumps(
                        texts, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "common_prefix_tokens": prefix,
                "queries": queries,
            }
        )
    frame = material.canonicalize_comments(
        pd.DataFrame(rows), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    bundle = material.read_factor_bundle(material.BUNDLE)
    _, _, scored = material.score_operator_from_comments(
        frame,
        spec=material.ObservationSpec(**bundle.operator),
        representation=joblib.load(material.REPRESENTATION),
        bundle=bundle,
    )
    if len(scored) != len(metadata) or not scored.score_status.eq("SCORED").all():
        raise ValueError(
            "All preselected slices must score under the frozen GRAIN operator"
        )
    scored = pd.DataFrame(metadata).merge(
        scored[["user_id", *material.FC]], on="user_id", validate="one_to_one"
    )
    coords = {}
    for a in design:
        i = a["index"]
        g = (
            scored[scored["index"] == i]
            .sort_values(["comment", "slice"])[material.FC]
            .to_numpy(float)
        )
        pos, token_g, w = material.allocate_slices(*allocations[i], g)
        coords[f"a{i}_positions"], coords[f"a{i}_grain"] = pos, token_g
        a.update(
            slices=len(g),
            selected_tokens=len(pos),
            allocation_error=float(abs(w.sum(1) - 1).max()),
        )
    out.mkdir(exist_ok=False)
    np.savez(out / "private_coordinates.npz", **coords)
    old.write(out / "private_design.json", design)
    old.write(out / "private_jobs.json", old.read(previous.OUT / "private_jobs.json"))
    inputs = dict(old.verify(previous.OUT)["inputs"])
    for folder in ("teachers", "models", "predictions", "analysis"):
        fr = ep.freeze(previous.OUT, folder)
        inputs.update(
            {
                str((previous.OUT / folder / n).relative_to(ROOT)): h
                for n, h in fr["files"].items()
            }
        )
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        path,
        source_path,
        eval_path,
        material.SOURCE / "manifest.json",
        material.BUNDLE,
        material.REPRESENTATION,
        ROOT / "scripts/run_grain_source_diversity_numeric.py",
        ROOT / "scripts/report_grain_source_diversity_teaching.py",
        ROOT / "tests/test_grain_source_diversity_teaching.py",
        material.MODEL / "config.json",
        material.MODEL / "tokenizer.json",
        material.MODEL / "tokenizer_config.json",
        Path(material.__file__),
        previous.OUT / "audit.json",
    ]
    for p in paths:
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    native = old.read(ep.ENDPOINT / "manifest.json")
    summary = {
        "new_source_authors": 8,
        "total_sources": 12,
        "recipient_authors": 8,
        "source_recipient_overlap": 0,
        "remaining_before_selection": counts,
        "unused_after_selection": 4,
        "comments": 96,
        "slices": sum(a["slices"] for a in design),
        "selected_tokens": sum(a["selected_tokens"] for a in design),
        "max_query_tokens": max(
            len(v["ids"]) for a in design for v in a["queries"].values()
        ),
        "human_labels_read": False,
        "native_calls": 0,
        "fits": 0,
    }
    old.write(out / "preparation_summary.json", summary)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_DIVERSITY_TEACHING_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {
                n: old.sha(out / n)
                for n in (
                    "private_design.json",
                    "private_coordinates.npz",
                    "private_jobs.json",
                    "preparation_summary.json",
                )
            },
            "prepare_cpu_seconds": 0.0,
            "prepare_wall_seconds": time.monotonic() - start,
            "selection_salt": fn.SALT,
            "max_cpu_seconds": CAP,
            "max_native_seconds": NATIVE_CAP,
            "model_sha256": native["model_sha256"],
            "alphabet": native["alphabet"],
            "c": native["c"],
            "native_limits": {
                "prefix_forwards": 16,
                "pre22_jvps": 64,
                "reverse_pairs": 16,
                "endpoint_jvps": 1,
                "fd_forwards": 4,
            },
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", **summary}, flush=True)


def native_packet(state, ds, layout, weights, qweights):
    """Reconstruct exactly the old native-Q, fixed-reader K/V local targets."""
    data = layout | {"state": state}
    keys, dk = ep.fn.value.scalar.identity_keys(data, ds, weights)
    v, dv = ep.fn.value.values(state, ds, weights)
    q = ep.source_query(state[-1], qweights)
    ell, dell = ep.fn.value.scalar.logits(q, keys, dk, data, float(weights["scale"]))
    return ep.fn.vectors(ell, dell, layout["mask"], v, dv, weights["o"])


def capture(out):
    """Capture eight complete source units under cost/validity-only monitoring."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.selection_endpoint import (
        endpoint,
        endpoint_gradients,
    )
    from grain_research.vector_mechanisms.upstream_attention_channels import (
        export_weights,
        pre_attention,
    )
    from scripts.run_grain_complete_transport_jvp import load_observer

    m = old.verify(out)
    dst = out / "capture"
    dst.mkdir(exist_ok=False)
    design = old.read(out / "private_design.json")
    coords = old.arrays(out / "private_coordinates.npz")
    ob, loading = load_observer(m)
    torch.set_grad_enabled(False)
    weights, qw = [
        old.arrays(root / "private_weights.npz") for root in (ep.STATE, ep.CAPTURE)
    ]
    actual = export_weights(ob.model)
    for k in qw:
        np.testing.assert_array_equal(qw[k], actual[k])
    limits = m["native_limits"]
    counters = {k: 0 for k in limits}
    counters["native_seconds"] = 0.0
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    c = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    names, checks = [], []
    start = time.monotonic()
    completed = 0

    def timed(kind, call):
        if counters[kind] >= limits[kind] or counters["native_seconds"] >= NATIVE_CAP:
            raise TimeoutError("NEXT88 source capture budget reached")
        torch.mps.synchronize()
        begin = time.monotonic()
        counters[kind] += 1
        try:
            return call()
        finally:
            torch.mps.synchronize()
            counters["native_seconds"] += time.monotonic() - begin

    def status(state):
        value = {
            "status": state,
            **counters,
            "complete_source_authors": completed,
            "loading_seconds": loading,
            "wall_seconds": time.monotonic() - start,
            "checks": checks,
            "files": {n: old.sha(dst / n) for n in names},
            "human_labels_read": False,
        }
        old.write(dst / "status.json", value)
        print(
            {k: v for k, v in value.items() if k not in ("files", "checks")}, flush=True
        )
        return value

    try:
        for a in design:
            i = a["index"]
            pos, g = coords[f"a{i}_positions"], coords[f"a{i}_grain"]
            common = None
            pair_states = []
            for qi, q in enumerate(old.QUERIES):
                raw, kw, _ = timed(
                    "prefix_forwards",
                    lambda a=a, q=q: material.capture_boundary(
                        ob, a["queries"][q]["ids"], a["queries"][q]["alphabet"]
                    ),
                )
                h = raw.detach().cpu().numpy()[0].copy()
                if common is None:
                    common = h.copy()
                prefix = a["common_prefix_tokens"]
                difference = float(
                    abs(h[:prefix].astype(float) - common[:prefix]).max()
                )
                if difference > 1e-4:
                    raise ValueError(
                        "Common-prefix precision differs materially from the frozen capture contract"
                    )
                h[:prefix] = common[:prefix]
                state = torch.tensor(h[None], device="mps", dtype=torch.float32)
                kw = cached_suffix_kwargs(ob.model, state)
                layout = {
                    k: t.detach().cpu().numpy()[0]
                    for k, t in zip(
                        ("cos", "sin"), kw["position_embeddings"], strict=True
                    )
                }
                layout["mask"] = kw["attention_mask"][0, 0, -1].detach().cpu().numpy()
                data = {"state": h, "grain": g, "positions": pos, **layout}
                ds = ep.body_directions(data)
                xs, vs = [], []
                for f in range(4):
                    d = torch.tensor(ds[f : f + 1], device="mps", dtype=torch.float32)
                    x, v = timed(
                        "pre22_jvps",
                        lambda state=state, kw=kw, d=d: torch.func.jvp(
                            lambda hh: pre_attention(ob.model, hh, kw), (state,), (d,)
                        ),
                    )
                    xs.append(x.detach().cpu().numpy()[0])
                    vs.append(v.detach().cpu().numpy()[0])
                primal_error = max(float(abs(x - xs[0]).max()) for x in xs)
                if primal_error > 1e-5:
                    raise ValueError("Repeated native pre22 primal mismatch")
                native_x, native_d = xs[0], np.stack(vs)
                pre = torch.tensor(native_x[None], device="mps", dtype=torch.float32)
                value, gx, r, post = timed(
                    "reverse_pairs",
                    lambda pre=pre, kw=kw: endpoint_gradients(
                        ob.model, pre, kw, ids, c
                    ),
                )
                packet = native_packet(native_x, native_d, layout, weights, qw)
                rr = r.cpu().numpy()[0].astype(float)
                checks.append(
                    {
                        "query_index": qi,
                        "author_index": i,
                        "common_prefix_max": difference,
                        "primal_max": primal_error,
                    }
                )
                if i == 0:
                    if qi == 0:
                        delta = torch.tensor(
                            native_d[0:1], device="mps", dtype=torch.float32
                        )
                        _, jvp = timed(
                            "endpoint_jvps",
                            lambda pre=pre, delta=delta, kw=kw: torch.func.jvp(
                                lambda hh: endpoint(ob.model, hh, kw, ids, c),
                                (pre,),
                                (delta,),
                            ),
                        )
                        expected = float(
                            np.sum(gx.cpu().numpy()[0].astype(float) * native_d[0])
                        )
                        np.testing.assert_allclose(
                            float(jvp.cpu()), expected, rtol=3e-4, atol=1e-5
                        )
                        checks[-1]["endpoint_jvp_absolute"] = abs(
                            float(jvp.cpu()) - expected
                        )
                    u = packet["vectors"][0, 0]
                    delta = torch.zeros_like(post)
                    delta[:, -1] = torch.tensor(u, device="mps", dtype=torch.float32)
                    vals = [
                        timed(
                            "fd_forwards",
                            lambda s=s, post=post, delta=delta, kw=kw: endpoint(
                                ob.model, post + s * 0.05 * delta, kw, ids, c, post=True
                            ),
                        )
                        for s in (-1, 1)
                    ]
                    fd = float(((vals[1] - vals[0]) / 0.1).cpu())
                    expected = float(rr @ u)
                    np.testing.assert_allclose(fd, expected, rtol=0.03, atol=2e-4)
                    checks[-1]["endpoint_fd_absolute"] = abs(fd - expected)
                name = f"a{i:02d}_{q}.npz"
                np.savez(
                    dst / name,
                    **data,
                    native_x=native_x,
                    native_d=native_d,
                    native_r=rr,
                    native_primal=packet["primal"],
                    native_vectors=packet["vectors"],
                )
                names.append(name)
                pair_states.append(data)
                del raw, state, pre, x, v, gx, r, post, value
                status("RUNNING")
            fn.previous.base.features(pair_states)
            completed += 1
            status("RUNNING")
            if completed == 2 and counters["native_seconds"] * 4 > NATIVE_CAP:
                status("PAUSED_RESOURCE_BOUNDARY")
                return
        if counters["native_seconds"] > NATIVE_CAP:
            raise TimeoutError("Budget exceeded after final native call")
        record = status("COMPLETE")
        record["files"]["status.json"] = old.sha(dst / "status.json")
        old.write(dst / "freeze.json", record)
    except Exception:
        status("FAILED")
        raise


def teachers(out):
    """Build the12-source teaching package from source-only native captures."""
    old.verify(out)
    ep.freeze(out, "capture")
    used, start = p85.cpu(out), time.process_time()
    dst = out / "teachers"
    dst.mkdir(exist_ok=False)
    paths = [
        out / "capture" / f"a{i:02d}_{q}.npz" for i in range(8) for q in old.QUERIES
    ]
    reads = [
        *paths,
        p85.BASELINE,
        previous.SOURCE_MODEL,
        previous.OUT / "teachers/private_teachers.npz",
        *p85.compiled_paths(),
    ]
    probes = [
        p85.OUT / "analysis/private_new8_a00.npz",
        p85.OUT / "predictions/new8_a00.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in ("private_teachers.npz", "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    b, m = old.arrays(p85.BASELINE), old.arrays(previous.SOURCE_MODEL)
    old_t = old.arrays(previous.OUT / "teachers/private_teachers.npz")
    compiler = p85.Compiler(16)
    points, cur, native, rr = [], [], [], []
    for i in range(8):
        pair = [old.arrays(out / "capture" / f"a{i:02d}_{q}.npz") for q in old.QUERIES]
        data = [
            {k: a[k] for k in ("state", "grain", "positions", "cos", "sin", "mask")}
            for a in pair
        ]
        p = compiler(data, [ep.body_directions(d) for d in data])
        points.append(
            fn.previous.base.flatten(
                fn.previous.base.features(data), b["scale"], "grain_field"
            )
        )
        cur.append(
            np.concatenate((p["primal"][:, 3, None], p["vectors"][:, 3, 2]), axis=1)
        )
        native.append(
            np.stack(
                [
                    np.concatenate(
                        (a["native_primal"][None], a["native_vectors"][2]), axis=0
                    )
                    for a in pair
                ]
            )
        )
        rr.append(np.stack([a["native_r"] for a in pair]))
        limit(start, used)
    current, native, rr = [
        np.concatenate((old_t[k], np.stack(new)), axis=0)
        for k, new in (("current", cur), ("native", native), ("native_readers", rr))
    ]
    t = fn.previous.teaching(current, native, rr, b["m"], b["d0"], m["units"])
    for k in t:
        np.testing.assert_array_equal(t[k][:4], old_t[k])
    np.savez(
        dst / "private_teachers.npz",
        **t,
        points=np.concatenate((m["x"][fn.previous.ORIGINAL], np.stack(points))),
        current=current,
        native=native,
        native_readers=rr,
    )
    ep.previous.finish(
        out,
        "teachers",
        ["private_teachers.npz"],
        start,
        compiled_query_calls=compiler.calls,
        source_authors=12,
        extra_denials=len(denied) - len(probes),
        native_calls=0,
    )


def models(out):
    """Two matched12-source value solves with previous source units and weight."""
    old.verify(out)
    ep.freeze(out, "teachers")
    used, start = p85.cpu(out), time.process_time()
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [v + ".npz" for v in fn.VARIANTS]
    probes = [
        p85.OUT / "analysis/private_new8_a00.npz",
        p85.OUT / "predictions/new8_a00.npz",
    ]
    denied = install(
        [previous.SOURCE_MODEL, out / "teachers/private_teachers.npz"],
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    m, t = (
        old.arrays(previous.SOURCE_MODEL),
        old.arrays(out / "teachers/private_teachers.npz"),
    )
    for n, k in zip(
        names, ("native_coordinates", "compatible_coordinates"), strict=True
    ):
        np.savez(dst / n, **fn.fit(m, t["points"], t[k]))
        limit(start, used)
    ep.previous.finish(
        out,
        "models",
        names,
        start,
        source_solves=2,
        coefficients_each=60,
        extra_denials=len(denied) - len(probes),
        human_labels_read=False,
    )


def main():
    """One stage per process; do not silently resume a partial capture."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "capture",
            "teachers",
            "models",
            "predict",
            "evaluate",
            "replay",
            "replay-evaluate",
            "audit",
        ),
    )
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    from scripts import run_grain_source_diversity_numeric as numeric

    with threadpool_limits(limits=1):
        if a.stage in ("predict", "replay"):
            numeric.predict(
                a.output, "predictions" if a.stage == "predict" else "replay"
            )
        elif a.stage in ("evaluate", "replay-evaluate"):
            numeric.evaluate(
                a.output, "predictions" if a.stage == "evaluate" else "replay"
            )
        else:
            {
                "prepare": prepare,
                "capture": capture,
                "teachers": teachers,
                "models": models,
                "audit": numeric.audit,
            }[a.stage](a.output)


if __name__ == "__main__":
    main()

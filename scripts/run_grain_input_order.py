#!/usr/bin/env python3
"""Prepare, acquire and analyze a bounded same-comment reversal experiment."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.configuration_extraction import InputProgram
from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.input_order import (
    fit_ordinal,
    predict_ordinal,
    reversal_account,
)
from grain_research.vector_mechanisms.natural_transport import (
    fit_operator,
    paired_summary,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_conditional_edits import LEVEL, load_panel
from scripts.run_grain_natural_transport import FACTORS
from scripts.run_grain_natural_transport import SOURCE as M29
from scripts.run_grain_reading_levels import load_inputs

PLAN = ROOT / "docs/GRAIN_LOGIC_INPUT_ORDER_PLAN_V1.md"
CORE = ROOT / "grain_research/vector_mechanisms/input_order.py"
S1 = BASE / "multiquery_sensitivity_s1_r1"
PRIOR = BASE / "logic_reading_levels_v1"
QUERIES = ("TF_cont", "neuroticism")


def source_fit(out, meta, bags, paths):
    """Freeze all source-fold programs and test predictions before endpoints/Y."""
    rows, held = [], []
    for q in QUERIES:
        group = "mbti_only" if q == "TF_cont" else "big5_only"
        d = meta[meta.group == group].sort_values(["author_id", "view"]).copy()
        d["feature_index"] = d.index
        d = d.merge(
            paths[paths["query"] == q][["author_id", "view", "configuration"]],
            on=["author_id", "view"],
            validate="one_to_one",
        )
        ids = d.author_id.unique()
        if len(d) != 3 * len(ids) or len(ids) != (63 if q == "TF_cont" else 213):
            raise ValueError("Wrong source population")
        xg = d[FACTORS].to_numpy().reshape(-1, 3, len(FACTORS))
        yc = d.configuration.to_numpy().reshape(-1, 3)
        splits = list(KFold(5, shuffle=True, random_state=20260910).split(ids))
        splits.append((np.arange(len(ids)), np.array([], int)))
        for fold, (train, test) in enumerate(splits):
            tr = np.concatenate([np.arange(i * 3, (i + 1) * 3) for i in train])
            data = d.iloc[tr]
            gm = fit_operator(xg[train], yc[train], "between")
            grain = data[FACTORS].to_numpy() @ gm["coef"] + gm["intercept"]
            target = np.column_stack([data.configuration, data.configuration - grain])
            xb = [bags[i] for i in data.feature_index]
            base = InputProgram("e5_mean", np.zeros((2, 1024))).fit(xb, None, target)
            new = fit_ordinal(xb, target, base)
            if fold < 5:
                te = np.concatenate([np.arange(i * 3, (i + 1) * 3) for i in test])
                part = d.iloc[te]
                tb = [bags[i] for i in part.feature_index]
                gt = part[FACTORS].to_numpy() @ gm["coef"] + gm["intercept"]
                observed = np.column_stack(
                    [part.configuration, part.configuration - gt]
                )
                for arm, pred in (
                    ("mean", base.predict(tb, None)),
                    ("ordinal", predict_ordinal(new, tb)),
                ):
                    for i, row in enumerate(part.itertuples()):
                        rows.append(
                            {
                                "author_id": row.author_id,
                                "view": row.view,
                                "query": q,
                                "fold": fold,
                                "arm": arm,
                                "actual_c": observed[i, 0],
                                "actual_r": observed[i, 1],
                                "predicted_c": pred[i, 0],
                                "predicted_r": pred[i, 1],
                            }
                        )
            else:
                joblib.dump(
                    {"program": new, "grain": gm, "train_authors": ids.tolist()},
                    out / f"private_{q}_program.joblib",
                )
                part = meta[(meta.group == "held") & (meta.edit == "full")].sort_values(
                    "author_id"
                )
                if set(part.author_id) & set(ids) or len(part) != 12:
                    raise ValueError("Held author identity")
                full = [bags[i] for i in part.index]
                reverse = [b[::-1] for b in full]
                for arm in ("mean", "ordinal"):
                    for order, xb in (("original", full), ("reverse", reverse)):
                        pred = (
                            base.predict(xb, None)
                            if arm == "mean"
                            else predict_ordinal(new, xb)
                        )
                        for i, uid in enumerate(part.author_id):
                            held.append(
                                {
                                    "author_id": uid,
                                    "query": q,
                                    "arm": arm,
                                    "order": order,
                                    "predicted_c": pred[i, 0],
                                    "predicted_r": pred[i, 1],
                                }
                            )
    pd.DataFrame(rows).to_csv(out / "private_source_oof.csv", index=False)
    pd.DataFrame(held).to_csv(out / "private_held_predictions.csv", index=False)
    old = pd.read_csv(PRIOR / "private_source_oof.csv").query("arm=='pooled'")
    new = pd.DataFrame(rows).query("arm=='mean'")
    match = new.merge(
        old,
        on=["author_id", "view", "query", "fold"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    error = max(
        float((match[k + "_new"] - match[k + "_old"]).abs().max())
        for k in ("actual_c", "actual_r", "predicted_c", "predicted_r")
    )
    if len(match) != 828 or error > 1e-10:
        raise ValueError("Prior source program changed")
    write(
        out / "prediction_freeze.json",
        {
            "old_rows": len(match),
            "old_max_error": error,
            "labels_read": False,
            "files": {p.name: sha(p) for p in out.glob("private_*.csv")},
        },
    )


def prepare(out):
    """Bind raw materials, cached vectors and old query masks with no outcome choice."""
    from transformers import AutoTokenizer

    from scripts.run_grain_level_response import question_mask

    native, states, lm, _ = load_panel()
    sm = read(S1 / "manifest.json")
    if sha(S1 / "private_material.json") != sm["material_sha256"]:
        raise ValueError("Held material changed")
    material = {
        r["author_id"]: r
        for r in read(S1 / "private_material.json")
        if r["view"] == 0 and r["author_id"] in set(native.author_id)
    }
    old = {(r["author_id"], r["query"]): r for r in read(LEVEL / "private_design.json")}
    model = Path(sm["model_dir"])
    tok = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    design = []
    functions = (
        Path(read(M29 / "manifest.json")["functions"]) / "nine_query_functions.npz"
    )
    with np.load(functions, allow_pickle=False) as z:
        fs = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in QUERIES
        }
    for i, uid in enumerate(sorted(material)):
        texts = material[uid]["texts"]
        if len(texts) != 12:
            raise ValueError("Require12 unchanged comments")
        for q in QUERIES:
            d = old[(uid, q)]
            ix = native.index[(native.author_id == uid) & (native["query"] == q)][0]
            inputs = {}
            for order, bodies in (("original", texts), ("reverse", texts[::-1])):
                _, _, enc, mask = question_mask(tok, bodies, q)
                ids = enc["input_ids"]
                if (
                    len(ids) + 16 > 3072
                    or list(np.asarray(ids)[mask]) != lm["question_tokens"][q]
                ):
                    raise ValueError("Query identity or length")
                inputs[order] = {"ids": ids, "mask": mask.tolist()}
            if inputs["original"]["ids"] != d["full_ids"]:
                raise ValueError("Original prompt replay changed")
            ca, cb = (
                Counter(inputs["original"]["ids"]),
                Counter(inputs["reverse"]["ids"]),
            )
            token_audit = {
                "length_delta": len(inputs["reverse"]["ids"])
                - len(inputs["original"]["ids"]),
                "token_bag_l1": sum(abs(ca[k] - cb[k]) for k in ca.keys() | cb.keys()),
                "comment_bag_unchanged": True,
            }
            design.append(
                dict(
                    index=i,
                    author_id=uid,
                    query=q,
                    **inputs,
                    token_audit=token_audit,
                    alphabet=d["alphabet"],
                    old_native=float(native.loc[ix, "teacher_full"]),
                    old_parts=fs[q].parts(states[ix]["full_P"], states[ix]["full_Q"]),
                )
            )
    write(out / "private_design.json", design)
    with threadpool_limits(limits=2):
        source_fit(out, *load_inputs())
    write(
        out / "manifest.json",
        {
            "model_dir": str(model),
            "model_sha256": lm["model_sha256"],
            "functions": str(functions),
            "functions_sha256": sha(functions),
            "plan_sha256": sha(PLAN),
            "code_hashes": {
                str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), CORE)
            },
            "design_sha256": sha(out / "private_design.json"),
            "max_calls": 52,
            "max_seconds": 600,
            "source_input_sha256": sha(
                BASE / "logic_configuration_extraction_v1/private_features.npz"
            ),
            "labels_used_for_fitting": False,
        },
    )
    write(
        out / "status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed": 0},
    )
    print(
        "Prepared24 paired units; source programs and held predictions frozen",
        flush=True,
    )


def acquire(out):
    """Run exactly declared full/reverse pairs, with complete-unit checkpoints."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, prior = read(out / "manifest.json"), read(out / "status.json")
    if prior["status"] == "COMPLETE":
        print("Acquisition complete; no new calls")
        return
    if (
        sha(out / "private_design.json") != m["design_sha256"]
        or sha(PLAN) != m["plan_sha256"]
    ):
        raise ValueError("Design changed")
    for name, digest in m["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Prepared code changed")
    if (
        sha(Path(m["model_dir"]) / "model.safetensors") != m["model_sha256"]
        or sha(m["functions"]) != m["functions_sha256"]
    ):
        raise ValueError("Frozen model/function changed")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        m["model_dir"], device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, old = observer.manifest(), read(LEVEL / "observer.json")
    for k in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[k] != old[k]:
            raise ValueError(f"Observer changed: {k}")
    write(out / "observer.json", current)
    with np.load(m["functions"], allow_pickle=False) as z:
        fs = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in QUERIES
        }
    start, complete = time.monotonic(), 0

    def status(state):
        v = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed": complete,
        }
        write(out / "status.json", v)
        print(v, flush=True)

    try:
        for d in read(out / "private_design.json"):
            target = (
                out / "private_packets" / f"unit_{d['index']:02d}_{d['query']}.json"
            )
            if target.exists():
                r = read(target)
                if (
                    r["author_id"] != d["author_id"]
                    or sha(target.with_suffix(".npz")) != r["arrays_sha256"]
                ):
                    raise ValueError("Cached completed packet changed")
                complete += 1
                continue
            needed = 3 if d["index"] < 2 else 2
            if (
                prior["calls"] + observer.calls + needed > m["max_calls"]
                or time.monotonic() - start + prior["seconds"] > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            values, arrays = {}, {}
            for order in ("original", "reverse"):
                h, kw, native = capture_boundary(
                    observer, d[order]["ids"], d["alphabet"]
                )
                p = h[0, d[order]["mask"]].cpu().numpy().astype(float)
                q = h[0, -1].cpu().numpy().astype(float)
                parts = fs[d["query"]].parts(p, q)
                values[order] = {"native": native["conditional_score"], "parts": parts}
                arrays.update({order + "_P": p, order + "_Q": q})
                del h, kw
            values["old_replay_error"] = abs(
                values["original"]["native"] - d["old_native"]
            )
            values["old_component_error"] = max(
                abs(values["original"]["parts"][k] - v)
                for k, v in d["old_parts"].items()
            )
            if max(values["old_replay_error"], values["old_component_error"]) > 2e-6:
                raise ValueError("Old native/function replay failed")
            if d["index"] < 2:
                h, kw, repeated = capture_boundary(
                    observer, d["original"]["ids"], d["alphabet"]
                )
                values["same_run_repeat_error"] = abs(
                    repeated["conditional_score"] - values["original"]["native"]
                )
                del h, kw
                if values["same_run_repeat_error"] > 2e-6:
                    raise ValueError("Native repeat instability")
            np.savez_compressed(target.with_suffix(".npz"), **arrays)
            write(
                target,
                dict(
                    author_id=d["author_id"],
                    query=d["query"],
                    index=d["index"],
                    arrays_sha256=sha(target.with_suffix(".npz")),
                    **values,
                ),
            )
            complete += 1
            gc.collect()
            torch.mps.empty_cache()
            status("RUNNING")
        status("COMPLETE")
    except Exception:
        status("FAILED")
        raise


def analyze(out):
    """Compare the changed target, invariant floor and frozen ordinal transport."""
    if read(out / "status.json")["status"] != "COMPLETE":
        raise ValueError("Need complete acquisition")
    for name, digest in read(out / "prediction_freeze.json")["files"].items():
        if sha(out / name) != digest:
            raise ValueError("Frozen prediction changed")
    rows = []
    for path in sorted((out / "private_packets").glob("unit_*.json")):
        r = read(path)
        if sha(path.with_suffix(".npz")) != r["arrays_sha256"]:
            raise ValueError("Packet changed")
        for order in ("original", "reverse"):
            rows.append(
                dict(
                    author_id=r["author_id"],
                    query=r["query"],
                    order=order,
                    native=r[order]["native"],
                    **r[order]["parts"],
                )
            )
    native = pd.DataFrame(rows)
    native.to_csv(out / "private_observations.csv", index=False)
    pred = pd.read_csv(out / "private_held_predictions.csv")
    accounts, response = [], []
    for q in QUERIES:
        a = (
            native[(native["query"] == q) & (native.order == "original")]
            .set_index("author_id")
            .sort_index()
        )
        b = (
            native[(native["query"] == q) & (native.order == "reverse")]
            .set_index("author_id")
            .loc[a.index]
        )
        inv = (
            pred[(pred["query"] == q) & (pred.arm == "mean")]
            .pivot(index="author_id", columns="order", values="predicted_c")
            .loc[a.index]
        )
        if np.max(np.abs(inv.original - inv.reverse)) > 1e-12:
            raise ValueError("Mean program not invariant")
        for component in ("configuration", "mean", "final", "score", "native"):
            # Only configuration has a trained matching invariant prediction.
            p = (
                inv.original
                if component == "configuration"
                else (a[component] + b[component]) / 2
            )
            accounts.append(
                dict(
                    query=q,
                    component=component,
                    **reversal_account(a[component], b[component], p),
                )
            )
        for arm in ("mean", "ordinal"):
            pp = (
                pred[(pred["query"] == q) & (pred.arm == arm)]
                .pivot(index="author_id", columns="order", values="predicted_c")
                .loc[a.index]
            )
            dp = (pp.reverse - pp.original).to_numpy()
            for endpoint in ("configuration", "native"):
                response.append(
                    dict(
                        query=q,
                        arm=arm,
                        endpoint=endpoint,
                        **paired_summary(
                            (b[endpoint] - a[endpoint]).to_numpy(),
                            dp,
                            calibrated=endpoint == "configuration",
                        ),
                    )
                )
        closure = (b.score - a.score) - sum(
            b[k] - a[k] for k in ("mean", "configuration", "final")
        )
        if np.max(np.abs(closure)) > 1e-12:
            raise ValueError("Fixed function component accounting failed")
    pd.DataFrame(accounts).to_csv(out / "orbit_accounts.csv", index=False)
    pd.DataFrame(response).to_csv(out / "response_metrics.csv", index=False)
    oof = pd.read_csv(out / "private_source_oof.csv")
    labels = pd.read_csv(M29 / "private_external_links.csv").set_index(
        ["author_id", "query"]
    )
    links = []
    for (q, arm), d in oof.groupby(["query", "arm"]):
        d = (
            d.groupby("author_id")[
                ["actual_c", "actual_r", "predicted_c", "predicted_r"]
            ]
            .mean()
            .sort_index()
        )
        y = labels.xs(q, level="query").loc[d.index].y_true
        for t in ("c", "r"):
            links.append(
                dict(
                    query=q,
                    arm=arm,
                    target=t,
                    endpoint="teacher",
                    **paired_summary(d["actual_" + t], d["predicted_" + t]),
                )
            )
            links.append(
                dict(
                    query=q,
                    arm=arm,
                    target=t,
                    endpoint="criterion",
                    **association(d["predicted_" + t], y, binary=q == "TF_cont"),
                )
            )
    pd.DataFrame(links).to_csv(out / "source_links.csv", index=False)
    write(
        out / "analysis_manifest.json",
        {
            "status": "COMPLETE",
            "labels_used_for_fitting": False,
            "packets": 24,
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "script_sha256": sha(__file__),
        },
    )
    print(pd.DataFrame(accounts).to_string(index=False), flush=True)
    print(pd.DataFrame(response).to_string(index=False), flush=True)


def main():
    """Keep preparation and endpoint acquisition independently inspectable."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run", "analyze"))
    p.add_argument("--output-dir", type=Path, default=BASE / "logic_input_order_v1")
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Use ignored private output directory")
    {"prepare": prepare, "run": acquire, "analyze": analyze}[a.stage](out)


if __name__ == "__main__":
    main()

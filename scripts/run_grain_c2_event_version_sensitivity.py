#!/usr/bin/env python
"""C2 event-version sensitivities, c2_event_version_v1 (docs/GRAIN_C2_EVENT_VERSION_SENSITIVITY_PROTOCOL_V1.md).

Stages: audit (provenance and alignment; no fits) | fit --view first|last (150 cells per view; resumable; capped).
The r1 runner, the dedup wrapper and the original MCD code are imported read-only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import importlib.util
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dd = _load("c2dedup", "scripts/run_grain_c2_dedup_sensitivity.py")
c2, orig = dd.c2, dd.orig
emo = __import__("scripts.run_situation_emotion_dynamics", fromlist=["x"])

VERSION = "c2_event_version_v1"
PROTOCOL = REPO / "docs/GRAIN_C2_EVENT_VERSION_SENSITIVITY_PROTOCOL_V1.md"
PROTOCOL_SHA = "677d32766380c092d0ffe4c4fcf0341c786e84cb34d0ca21faf7c694e0b9687b"
OUT = REPO / "results/grain_gptpro_foundation_v3/P05/c2_event_version_sensitivity_r1"
KEY = ["user_id", "comment_id"]
BODY_FEATS = ["token_count", "char_count", *dd.bench_cols.RESPONSE_COLS]
CAP_S = 90 * 60
ARMS = ("A", "B", "C")


def select_view(comments: pd.DataFrame, view: str) -> pd.DataFrame:
    return comments.drop_duplicates(KEY, keep="first" if view == "first" else "last").reset_index(drop=True)


def load_all() -> pd.DataFrame:
    artifacts = orig.load_baseline_artifacts(c2.BASELINE)
    return orig.load_comments(str(c2.COMMENTS), set(artifacts["unames"]), c2.original_args().version)


def content_hash(df: pd.DataFrame, keys: pd.Series, mask: np.ndarray) -> str:
    sub = df.loc[mask, dd.CONSUMED]
    k = keys[mask]
    rows = sorted(f"{kk}|" + hashlib.sha256(json.dumps([repr(v) for v in r], ensure_ascii=False).encode()).hexdigest()
                  for kk, r in zip(k, sub.itertuples(index=False, name=None)))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def audit() -> dict:
    if c2.sha256_file(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    allc = load_all()
    stored = allc[BODY_FEATS].to_numpy(float)
    recomputed = pd.DataFrame([emo.comment_emotion_features(t) for t in allc["body_raw"]])[BODY_FEATS].to_numpy(float)
    bad_row = np.abs(stored - recomputed).max(axis=1) > 1e-9
    dup_all = allc.duplicated(KEY, keep=False)
    out = {"version": VERSION, "rows_input": int(len(allc)), "keys_with_multiple_versions": int(allc[dup_all].groupby(KEY).ngroups),
           "rows_body_feature_inconsistent_all": int(bad_row.sum()),
           "inconsistent_columns_all": {c: int((np.abs(stored[:, i] - recomputed[:, i]) > 1e-9).sum()) for i, c in enumerate(BODY_FEATS) if (np.abs(stored[:, i] - recomputed[:, i]) > 1e-9).any()},
           "views": {}}
    for view in ("first", "last"):
        keep = "first" if view == "first" else "last"
        sel = ~allc.duplicated(KEY, keep=keep)
        vdf = allc[sel]
        other = allc[dup_all & ~sel]
        # provenance: for multi-version keys, how the selected record differs from the other versions
        firsts = vdf.set_index(KEY)
        prov = {"score_differs": 0, "word_count_differs": 0, "body_differs": 0}
        for r in other.itertuples(index=False):
            f = firsts.loc[(r.user_id, r.comment_id)]
            prov["score_differs"] += int(not (f["score"] == r.score or (pd.isna(f["score"]) and pd.isna(r.score))))
            prov["word_count_differs"] += int(f["word_count"] != r.word_count)
            prov["body_differs"] += int(f["body_raw"] != r.body_raw)
        trait = vdf["subreddit"].map(c2.is_trait_named)
        out["views"][view] = {"rows": int(sel.sum()), "selected_rows_body_feature_inconsistent": int(bad_row[sel.to_numpy()].sum()),
                              "trait_named_rows": int(trait.sum()), "authors": int(vdf.user_id.nunique()),
                              "dropped_other_version_rows": int(len(other)), "dropped_vs_selected_provenance": prov}
    ok = all(v["selected_rows_body_feature_inconsistent"] == 0 for v in out["views"].values())
    out["status"] = "PASS" if ok else "STOP_ALIGNMENT"
    (OUT / "version_inventory.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


class ViewContext(c2.Context):
    def __init__(self, view: str) -> None:
        self.view = view
        self.args = c2.original_args()
        self.artifacts = orig.load_baseline_artifacts(c2.BASELINE)
        comments = select_view(orig.load_comments(str(c2.COMMENTS), set(self.artifacts["unames"]), self.args.version), view)
        self.comments = comments
        self.keys = c2.row_keys(comments)
        self.masks, self.deletions = c2.build_arm_masks(comments)
        self.arm_comments = {a: comments.loc[self.masks[a]].reset_index(drop=True) for a in ARMS}
        self.input_hash = {a: content_hash(comments, self.keys, self.masks[a]) for a in ARMS}
        self.retained = {a: self.arm_comments[a]["user_id"].value_counts() for a in ARMS}
        self.cache_dir = OUT / f"mcd_cache_{view}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.frames = {}


def get_mcd(ctx: ViewContext, arm, tr, va, te, fold):
    payload = {"version": VERSION, "view": ctx.view, "arm": arm, "content_hash": ctx.input_hash[arm], "train": tr, "val": va, "test": te,
               **c2.mcd_settings(ctx.args)}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    npz, meta = ctx.cache_dir / f"{key}.npz", ctx.cache_dir / f"{key}.json"
    if npz.exists() and meta.exists():
        d = np.load(npz)
        return d["train"], d["val"], d["test"], json.loads(meta.read_text())["diag"], key, 0.0
    t0 = time.time()
    a, b, c, diag = orig.fit_fold_mcd(ctx.arm_comments[arm], tr, va, te, ctx.args, seed=ctx.args.random_seed + int(fold) * 1000)
    np.savez(npz, train=a, val=b, test=c)
    meta.write_text(json.dumps({"view": ctx.view, "arm": arm, "fold": fold, "seconds": time.time() - t0, "diag": orig.jsonable(diag)}) + "\n")
    return a, b, c, orig.jsonable(diag), key, time.time() - t0


def run_cell(ctx, condition, trait, fold, arm):
    spec = orig.CONDITION_SPECS[condition]
    frame, _v, p_matrix = ctx.frame(condition, trait)
    y = frame["label"].to_numpy(float)
    tri, vai, tei = orig.split_indices(frame, fold)
    au = frame["author"].astype(str).tolist()
    tr_u, va_u, te_u = [au[i] for i in tri], [au[i] for i in vai], [au[i] for i in tei]
    t0 = time.time()
    tr, va, te, diag, key, mcd_s = get_mcd(ctx, arm, tr_u, va_u, te_u, fold)
    full = np.zeros((len(frame), tr.shape[1]))
    full[tri], full[vai], full[tei] = tr, va, te
    dense = np.hstack([p_matrix, full]) if spec["use_p"] else full
    xtr, xva, xte, total = orig.build_design_matrices(ctx.artifacts, frame, tri, vai, tei, use_ngram=bool(spec["use_ngram"]), dense_extra=dense)
    pred, best, solvers = dd.fit_select_predict(xtr, y[tri], xva, y[vai], xte, condition=condition, trait=trait, fold=fold)
    ret = ctx.retained[arm]
    return {"condition": condition, "trait": trait, "fold": fold, "arm": arm + "'", "view": ctx.view, "version": VERSION,
            "pearson": orig.safe_pearson(y[tei], pred), "n_train": len(tri), "n_val": len(vai), "n_test": len(tei), "total_features": int(total),
            **best, "solvers": solvers, "mcd_cache_key": key, "mcd_fit_seconds": mcd_s, "cell_seconds": time.time() - t0,
            "n_unobserved_test": int(sum(ret.get(u, 0) == 0 for u in te_u)), "n_unobserved_train": int(sum(ret.get(u, 0) == 0 for u in tr_u)),
            "diag": diag, "predictions": {"author": te_u, "true": [float(v) for v in y[tei]], "pred": [float(v) for v in pred],
                                          "mcd_unobserved": [int(ret.get(u, 0) == 0) for u in te_u]}}


def fit(view: str) -> dict:
    inv = json.loads((OUT / "version_inventory.json").read_text())
    if inv["status"] != "PASS":
        raise RuntimeError("alignment audit did not pass; fits refused")
    bpath = OUT / "budget.json"
    budget = json.loads(bpath.read_text()) if bpath.exists() else {"packet_start_epoch": time.time()}
    bpath.write_text(json.dumps(budget) + "\n")
    start = budget["packet_start_epoch"]
    ctx = ViewContext(view)
    ctx.deletions.to_csv(OUT / f"deletion_counts_by_author_{view}.csv", index=False)
    man = {"view": view, "rows": {a: int(ctx.masks[a].sum()) for a in ARMS}, "content_hash": ctx.input_hash,
           "authors_unobserved_BC": int(ctx.deletions["mcd_unobserved_BC"].sum()),
           "authors_with_any_trait_row": int((ctx.deletions["n_removed_B"] > 0).sum()),
           "runner_sha256": c2.sha256_file(Path(__file__)), "protocol_sha256": PROTOCOL_SHA}
    (OUT / f"selection_manifest_{view}.json").write_text(json.dumps(man, indent=1) + "\n")
    cdir = OUT / f"cells_{view}"
    cdir.mkdir(exist_ok=True)
    log = (OUT / "progress.jsonl").open("a")
    times, stopped = [], None
    for condition, trait, fold, arm in c2.cell_plan():
        path = cdir / f"{condition}__{trait}__f{fold}__{arm}p.json"
        if path.exists():
            continue
        if CAP_S - 600 - (time.time() - start) < 1.5 * (max(times) if times else 30.0):
            stopped = {"reason": "budget", "next": [view, condition, trait, fold, arm]}
            break
        try:
            res = run_cell(ctx, condition, trait, fold, arm)
        except Exception as exc:
            log.write(json.dumps({"event": "cell_failed", "view": view, "cell": [condition, trait, fold, arm], "error": repr(exc), "tb": traceback.format_exc()}) + "\n")
            log.flush()
            continue
        path.write_text(json.dumps(orig.jsonable(res)) + "\n")
        times.append(res["cell_seconds"])
        log.write(json.dumps({"event": "cell_done", "view": view, "cell": [condition, trait, fold, arm + "'"], "r": res["pearson"], "solvers": res["solvers"], "elapsed": time.time() - start}) + "\n")
        log.flush()
    done = len([p for p in cdir.glob("*.json") if not p.name.startswith("._")])
    summ = {"view": view, "stopped": stopped, "complete_cells": done, "total_cells": 150, "elapsed_since_packet_start_s": time.time() - start,
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    (OUT / f"fit_summary_{view}.json").write_text(json.dumps(summ, indent=1) + "\n")
    return summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "fit"], required=True)
    ap.add_argument("--view", choices=["first", "last"])
    a = ap.parse_args()
    out = audit() if a.stage == "audit" else fit(a.view)
    print(json.dumps(out, indent=1)[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

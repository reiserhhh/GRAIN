#!/usr/bin/env python
"""C2 dedup sensitivity packet, version c2_dedup_v1 (docs/GRAIN_C2_DEDUP_SENSITIVITY_PROTOCOL_V1.md).

This wrapper imports the r1 C2 runner and the original MCD functions read-only. The r1 code, caches and outputs are
never modified.

Stages:
  inventory   duplicate-key equivalence check and condition-specific role eligibility; stops on a conflict
  fit         A'/B'/C' cells (only after an inventory PASS); resumable; one-hour cap
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import hashlib
import importlib.util
import json
import resource
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c2", REPO / "scripts/run_grain_c2_mcd_venue_dependence.py")
c2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c2)
orig = c2.orig
bench_cols = __import__("scripts.run_context_fixed_deviation_benchmark", fromlist=["x"])

VERSION = "c2_dedup_v1"
PROTOCOL = REPO / "docs/GRAIN_C2_DEDUP_SENSITIVITY_PROTOCOL_V1.md"
PROTOCOL_SHA = "48c15bc8891abbe1b0921417be2da2e017de33c426c31877fcfbf6b98d67710f"
OUT = REPO / "results/grain_gptpro_foundation_v3/P05/c2_dedup_sensitivity_r1"
R1 = REPO / "results/grain_gptpro_foundation_v3/P05/c2_venue_dependence_r1"
KEY = ["user_id", "comment_id"]
CONSUMED = ["user_id", "comment_id", "body_raw", "subreddit", *bench_cols.CONTEXT_NUMERIC_COLS, *bench_cols.RESPONSE_COLS]
BUDGET_S = 3600
ARMS = ("A", "B", "C")


def vhash(v) -> str:
    return hashlib.sha256(("<NaN>" if (isinstance(v, float) and np.isnan(v)) else repr(v)).encode()).hexdigest()[:16]


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def inventory() -> dict:
    raw = pd.read_csv(c2.COMMENTS)
    raw["user_id"] = raw["user_id"].astype(str)
    cols = list(raw.columns)
    unused = [c for c in cols if c not in CONSUMED]
    dup_mask = raw.duplicated(KEY, keep=False)
    groups = raw[dup_mask].groupby(KEY, sort=False)
    classes = {"exact": 0, "consumed_equivalent": 0, "conflicting": 0}
    diff_cols_consumed: dict[str, int] = {}
    diff_cols_unused: dict[str, int] = {}
    diff_examples: list[dict] = []
    extra_rows = 0
    for key, g in groups:
        first = g.iloc[0]
        for _, row in g.iloc[1:].iterrows():
            extra_rows += 1
            dc = [c for c in cols if not same(first[c], row[c])]
            dcons = [c for c in dc if c in CONSUMED]
            duse = [c for c in dc if c not in CONSUMED]
            for c in dcons:
                diff_cols_consumed[c] = diff_cols_consumed.get(c, 0) + 1
            for c in duse:
                diff_cols_unused[c] = diff_cols_unused.get(c, 0) + 1
            if dcons:
                classes["conflicting"] += 1
                if len(diff_examples) < 20:
                    diff_examples.append({"key_hash": vhash(key), "columns": dcons,
                                          "value_hashes": {c: [vhash(first[c]), vhash(row[c])] for c in dcons}})
            elif duse:
                classes["consumed_equivalent"] += 1
            else:
                classes["exact"] += 1
    # condition-specific eligibility
    artifacts = orig.load_baseline_artifacts(c2.BASELINE)
    comment_users = set(raw["user_id"])
    elig = {}
    for cond, use_p in (("np_mcd", True), ("mcd_only", False)):
        per_trait = {}
        for trait in orig.BIG5_TRAITS:
            frame, _v, _p = orig.trait_frame(artifacts, trait, 0, use_p)
            fa = set(frame["author"].astype(str))
            out_auth = sorted(comment_users - fa)
            per_trait[trait] = {"frame_authors": len(fa), "comment_authors_outside_frame": len(out_auth),
                                "rows_outside_frame": int(raw["user_id"].isin(out_auth).sum()),
                                "frame_authors_without_comments": len(fa - comment_users)}
        elig[cond] = per_trait
    # reason for np exclusions: missing P features
    pdf = artifacts["p_features"]
    p_ok = set(pdf.loc[np.isfinite(pdf[orig.P_FEATURES].to_numpy(float)).all(axis=1), "author"].astype(str))
    np_out = comment_users - set(orig.trait_frame(artifacts, "agreeableness", 0, True)[0]["author"].astype(str))
    mo_in = set(orig.trait_frame(artifacts, "agreeableness", 0, False)[0]["author"].astype(str))
    reasons = {"np_out_authors": len(np_out), "np_out_missing_P": len(np_out - p_ok),
               "np_out_with_P_but_other_reason": len(np_out & p_ok), "np_out_inside_mcd_only_frame": len(np_out & mo_in)}
    # r1 evaluation author sets per condition and trait (must be reproduced)
    r1_preds = pd.read_csv(R1 / "predictions.csv")
    r1_sets = {f"{c}|{t}": sorted(set(g.author.astype(str))) for (c, t), g in r1_preds[r1_preds.arm == "A"].groupby(["condition", "trait"])}
    same_sets = {}
    for cond, use_p in (("np_mcd", True), ("mcd_only", False)):
        for trait in orig.BIG5_TRAITS:
            fa = sorted(orig.trait_frame(artifacts, trait, 0, use_p)[0]["author"].astype(str))
            same_sets[f"{cond}|{trait}"] = fa == r1_sets[f"{cond}|{trait}"]
    dedup = raw.drop_duplicates(KEY, keep="first")
    trait_named = raw["subreddit"].map(c2.is_trait_named)
    dupx = raw.duplicated(KEY, keep="first")
    mult = {"extra_duplicate_rows": int(dupx.sum()), "keys_with_duplicates": int(groups.ngroups),
            "authors_with_duplicates": int(raw.loc[dupx, "user_id"].nunique()),
            "extra_rows_in_trait_named": int((dupx & trait_named).sum()),
            "trait_named_rows_before": int(trait_named.sum()), "trait_named_rows_after": int(raw.loc[~dupx, "subreddit"].map(c2.is_trait_named).sum()),
            "max_copies_per_key": int(raw.groupby(KEY).size().max()), "rows_before": int(len(raw)), "rows_after": int(len(dedup))}
    passed = classes["conflicting"] == 0 and all(same_sets.values())
    out = {"version": VERSION, "protocol_sha256": PROTOCOL_SHA, "input_sha256": c2.sha256_file(c2.COMMENTS),
           "consumed_fields": CONSUMED, "unused_fields": unused, "classes_of_extra_rows": classes,
           "differing_consumed_columns": diff_cols_consumed, "differing_unused_columns": diff_cols_unused,
           "conflict_examples_hashed": diff_examples, "multiplicities": mult, "eligibility": elig,
           "np_exclusion_reasons": reasons, "r1_evaluation_sets_reproduced": same_sets,
           "status": "PASS" if passed else "STOP_INVENTORY"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


class DedupContext(c2.Context):
    def __init__(self) -> None:  # noqa: D401 - same fields as c2.Context, deduplicated input
        self.args = c2.original_args()
        self.artifacts = orig.load_baseline_artifacts(c2.BASELINE)
        comments = orig.load_comments(str(c2.COMMENTS), set(self.artifacts["unames"]), self.args.version)
        comments = comments.drop_duplicates(KEY, keep="first").reset_index(drop=True)
        self.comments = comments
        self.keys = c2.row_keys(comments)
        self.masks, self.deletions = c2.build_arm_masks(comments)
        self.arm_comments = {arm: comments.loc[self.masks[arm]].reset_index(drop=True) for arm in ARMS}
        self.input_hash = {arm: c2.retained_hash(self.keys, self.masks[arm]) for arm in ARMS}
        self.retained = {arm: self.arm_comments[arm]["user_id"].value_counts() for arm in ARMS}
        self.cache_dir = OUT / "mcd_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.frames = {}


def get_mcd(ctx: DedupContext, arm: str, train, val, test, fold: int):
    payload = {"version": VERSION, "arm": arm, "input_hash": ctx.input_hash[arm], "train": train, "val": val, "test": test,
               **c2.mcd_settings(ctx.args)}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    npz, meta = ctx.cache_dir / f"{key}.npz", ctx.cache_dir / f"{key}.json"
    if npz.exists() and meta.exists():
        d = np.load(npz)
        return d["train"], d["val"], d["test"], json.loads(meta.read_text())["diag"], key, 0.0
    t0 = time.time()
    tr, va, te, diag = orig.fit_fold_mcd(ctx.arm_comments[arm], train, val, test, ctx.args, seed=ctx.args.random_seed + int(fold) * 1000)
    dt = time.time() - t0
    np.savez(npz, train=tr, val=va, test=te)
    meta.write_text(json.dumps({"arm": arm, "fold": fold, "seconds": dt, "diag": orig.jsonable(diag)}) + "\n")
    return tr, va, te, orig.jsonable(diag), key, dt


def fit_select_predict(x_train, y_train, x_val, y_val, x_test, *, condition, trait, fold):
    """r1's selection loop with the identical Ridge call; records the actual solver dispatched."""
    best_score, best_pred, best, solvers = -1e9, None, {}, set()
    for feat_n in orig.feature_selection_candidates(int(x_train.shape[1])):
        selector = SelectKBest(f_regression, k=feat_n)
        xtr = sparse.csr_matrix(selector.fit_transform(x_train, y_train))
        xva = sparse.csr_matrix(selector.transform(x_val))
        xte = sparse.csr_matrix(selector.transform(x_test))
        xtr.sort_indices()
        xva.sort_indices()

        def eval_alpha(alpha, feat_n=feat_n, xtr=xtr, xva=xva, xte=xte):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = Ridge(alpha=alpha, max_iter=20000, solver="sag", random_state=c2.paired_seed(condition, trait, fold, feat_n, alpha))
                model.fit(xtr, y_train)
            return orig.safe_pearson(y_val, model.predict(xva)), alpha, np.asarray(model.predict(xte), float), model.solver_

        for score, alpha, pred, solver in Parallel(n_jobs=2)(delayed(eval_alpha)(a) for a in c2.ALPHAS):
            solvers.add(solver)
            if np.isfinite(score) and score > best_score:
                best_score, best_pred = float(score), pred
                best = {"best_val_pearson": float(score), "best_alpha": float(alpha), "best_feat_n": str(feat_n)}
    return best_pred, best, sorted(solvers)


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
    pred, best, solvers = fit_select_predict(xtr, y[tri], xva, y[vai], xte, condition=condition, trait=trait, fold=fold)
    ret = ctx.retained[arm]
    return {"condition": condition, "trait": trait, "fold": fold, "arm": arm + "'", "version": VERSION,
            "pearson": orig.safe_pearson(y[tei], pred), "n_train": len(tri), "n_val": len(vai), "n_test": len(tei),
            "total_features": int(total), **best, "solvers": solvers, "mcd_cache_key": key, "mcd_fit_seconds": mcd_s,
            "cell_seconds": time.time() - t0, "n_unobserved_test": int(sum(ret.get(u, 0) == 0 for u in te_u)),
            "n_unobserved_train": int(sum(ret.get(u, 0) == 0 for u in tr_u)), "diag": diag,
            "predictions": {"author": te_u, "true": [float(v) for v in y[tei]], "pred": [float(v) for v in pred],
                            "mcd_unobserved": [int(ret.get(u, 0) == 0) for u in te_u]},
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}


def fit_stage() -> dict:
    inv = json.loads((OUT / "inventory.json").read_text())
    if inv["status"] != "PASS":
        raise RuntimeError("inventory did not pass; fit refused")
    start = time.time()
    (OUT / "budget.json").write_text(json.dumps({"fit_start_epoch": start, "fit_start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    ctx = DedupContext()
    ctx.deletions.to_csv(OUT / "deletion_counts_by_author.csv", index=False)
    manifest = {"version": VERSION, "rows": {a: int(ctx.masks[a].sum()) for a in ARMS}, "input_hash": ctx.input_hash,
                "authors_unobserved_BC": int(ctx.deletions["mcd_unobserved_BC"].sum()),
                "authors_with_any_trait_row": int((ctx.deletions["n_removed_B"] > 0).sum()),
                "runner_sha256": c2.sha256_file(Path(__file__)), "r1_runner_sha256": c2.sha256_file(REPO / "scripts/run_grain_c2_mcd_venue_dependence.py"),
                "original_file_sha256": {str(p.name): c2.sha256_file(p) for p in c2.ORIG_FILES}}
    (OUT / "inputs_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    cells_dir = OUT / "cells"
    cells_dir.mkdir(exist_ok=True)
    log = (OUT / "progress.jsonl").open("a")
    times: list[float] = []
    stopped = None
    for condition, trait, fold, arm in c2.cell_plan():
        path = cells_dir / f"{condition}__{trait}__f{fold}__{arm}p.json"
        if path.exists():
            continue
        remaining = BUDGET_S - 600 - (time.time() - start)  # 10 min reserved for analysis
        if remaining < 1.5 * (max(times) if times else 30.0):
            stopped = {"reason": "budget", "next": [condition, trait, fold, arm]}
            break
        try:
            res = run_cell(ctx, condition, trait, fold, arm)
        except Exception as exc:
            log.write(json.dumps({"event": "cell_failed", "cell": [condition, trait, fold, arm], "error": repr(exc), "tb": traceback.format_exc()}) + "\n")
            log.flush()
            continue
        path.write_text(json.dumps(orig.jsonable(res)) + "\n")
        times.append(res["cell_seconds"])
        log.write(json.dumps({"event": "cell_done", "cell": [condition, trait, fold, arm + "'"], "r": res["pearson"], "s": res["cell_seconds"], "solvers": res["solvers"], "elapsed": time.time() - start}) + "\n")
        log.flush()
    done = sorted(p.name for p in cells_dir.glob("*.json") if not p.name.startswith("._"))
    summary = {"stopped": stopped, "complete_cells": len(done), "total_cells": 150, "elapsed_seconds": time.time() - start}
    (OUT / "fit_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> int:
    if c2.sha256_file(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inventory", "fit"], required=True)
    stage = ap.parse_args().stage
    out = inventory() if stage == "inventory" else fit_stage()
    print(json.dumps({k: v for k, v in out.items() if k not in ("conflict_examples_hashed", "eligibility")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

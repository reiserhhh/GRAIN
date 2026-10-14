#!/usr/bin/env python
"""C2 paired MCD venue-dependence packet, version c2_v1 (seminar-06 item 2).

Protocol: docs/GRAIN_C2_MCD_VENUE_DEPENDENCE_PROTOCOL_V1.md, frozen before any fit.

The original MCD integration functions are imported read-only from the coordination
repo. The old runner, archived outputs and code are never modified. The only estimator
change is an explicit paired SAG ``random_state``: a labelled deterministic
replication variant, not a bit-identical Windows reproduction.

Stages:
  replay  A / np_mcd / agreeableness / fold 0 versus the archive, with declared gates
  full    all remaining cells (only after a replay PASS); resumable, budget-capped
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import hashlib
import json
import re
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge

PERSONA = Path("/Volumes/mobile3/projects/project persona")
REPO = Path(__file__).resolve().parents[1]
if str(PERSONA) not in sys.path:
    sys.path.insert(0, str(PERSONA))
import scripts.run_pandora_baseline_mcd_integration as orig  # noqa: E402

VERSION = "c2_v1"
PROTOCOL = REPO / "docs/GRAIN_C2_MCD_VENUE_DEPENDENCE_PROTOCOL_V1.md"
PROTOCOL_SHA = "9a4a4e3af33587a1f88dae95fddbab7d4e6ae528253cea0c9e8e154ff21a91c2"
AMENDMENT = REPO / "docs/GRAIN_C2_MCD_VENUE_DEPENDENCE_PROTOCOL_V1_A1.md"
AMENDMENT_SHA = "3d3b75ef962cade12b04a09e75b482e6f5ff5ce15188fe05c0c5c45d897ff0ce"
OUT = REPO / "results/grain_gptpro_foundation_v3/P05/c2_venue_dependence_r1"
ARCH = PERSONA / "results/pandora_baseline_mcd_integration"
COMMENTS = PERSONA / "analysis/output/situation_emotion_dynamics/20260613_041057/comment_emotion_features.csv"
BASELINE = PERSONA / "data_sets/PANDORA_official/pandora_baseline"
ORIG_FILES = [
    PERSONA / "scripts/run_pandora_baseline_mcd_integration.py",
    PERSONA / "scripts/run_context_fixed_deviation_benchmark.py",
    COMMENTS,
]
DELETION_SEED = "20260926"
TYPE_RE = re.compile(r"^[ie][ns][tf][jp]s?$")
SUBSTRINGS = ("mbti", "personality", "introvert", "extrovert", "extravert", "enneagram", "socionic", "typology", "typeme")
ARMS = ("A", "B", "C")
CONDITIONS = ("np_mcd", "mcd_only")
N_JOBS = 2
BUDGET_S = 4 * 3600
STAGE1_CAP_S = 30 * 60
ALPHAS = [2**idx for idx in range(-10, 5)]

EXACT_DIAG = [
    "train_comment_count", "target_comment_count", "comment_text_features", "context_metadata_feature_count",
    "mcd_k_actual", "mcd_feature_dim", "comment_svd_dims", "context_metadata_dims",
]
EXACT_METRIC = ["n_train", "n_val", "n_test", "total_features"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_trait_named(subreddit: Any) -> bool:
    name = str(subreddit).strip().lower()
    return bool(TYPE_RE.match(name)) or any(token in name for token in SUBSTRINGS)


def row_keys(comments: pd.DataFrame) -> pd.Series:
    occ = comments.groupby(["user_id", "comment_id"], sort=False).cumcount()
    return comments["user_id"].astype(str) + "|" + comments["comment_id"].astype(str) + "|" + occ.astype(str)


def build_arm_masks(comments: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return keep-masks for arms A/B/C and a per-author deletion table."""
    keys = row_keys(comments)
    if keys.duplicated().any():
        raise ValueError("row keys are not unique")
    trait = comments["subreddit"].fillna("__missing__").map(is_trait_named).to_numpy(bool)
    users = comments["user_id"].astype(str)
    d_u = pd.Series(trait, index=comments.index).groupby(users).sum()
    rank_key = keys.map(lambda k: hashlib.sha256(f"{DELETION_SEED}|{k}".encode("utf-8")).hexdigest())
    order = pd.DataFrame({"user": users, "h": rank_key}).sort_values(["user", "h"], kind="mergesort")
    rank = order.groupby("user", sort=False).cumcount().reindex(comments.index)
    keep_c = rank.to_numpy() >= users.map(d_u).to_numpy()
    masks = {"A": np.ones(len(comments), dtype=bool), "B": ~trait, "C": keep_c}
    table = pd.DataFrame({
        "author": d_u.index,
        "n_rows_A": users.value_counts().reindex(d_u.index).to_numpy(),
        "n_removed_B": d_u.to_numpy(int),
        "n_removed_C": (~pd.Series(keep_c, index=comments.index)).groupby(users).sum().reindex(d_u.index).to_numpy(int),
        "n_trait_rows_removed_by_C": pd.Series(trait & ~keep_c, index=comments.index).groupby(users).sum().reindex(d_u.index).to_numpy(int),
    })
    table["retained_B"] = table["n_rows_A"] - table["n_removed_B"]
    table["retained_C"] = table["n_rows_A"] - table["n_removed_C"]
    if not (table["retained_B"] == table["retained_C"]).all():
        raise AssertionError("B/C retained counts differ for some author")
    table["mcd_unobserved_BC"] = table["retained_B"] == 0
    return masks, table


def retained_hash(keys: pd.Series, mask: np.ndarray) -> str:
    return sha256_text("\n".join(sorted(keys[mask].tolist())))


def paired_seed(condition: str, trait: str, fold: int, feat_n: Any, alpha: float) -> int:
    digest = sha256_text(f"{VERSION}|{condition}|{trait}|{fold}|{feat_n}|{alpha!r}")
    return int(digest[:8], 16) % 2147483647


def mcd_cache_key(arm: str, input_hash: str, train: list[str], val: list[str], test: list[str], settings: dict[str, Any]) -> str:
    payload = {"version": VERSION, "arm": arm, "input_hash": input_hash, "train": train, "val": val, "test": test, **settings}
    return sha256_text(json.dumps(payload, sort_keys=True))


def original_args() -> argparse.Namespace:
    saved = sys.argv
    try:
        sys.argv = ["run_pandora_baseline_mcd_integration.py"]
        return orig.parse_args()
    finally:
        sys.argv = saved


def mcd_settings(args: argparse.Namespace) -> dict[str, Any]:
    keys = ["mcd_k", "mcd_tau_scale", "mcd_eps", "mcd_feature", "comment_max_features", "comment_svd_dims", "min_df", "random_seed", "version"]
    return {key: getattr(args, key) for key in keys}


def get_mcd(arm: str, arm_comments: pd.DataFrame, input_hash: str, train: list[str], val: list[str], test: list[str],
            fold: int, args: argparse.Namespace, cache_dir: Path, events: list[dict[str, Any]]):
    key = mcd_cache_key(arm, input_hash, train, val, test, mcd_settings(args))
    npz, meta = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
    if npz.exists() and meta.exists():
        data = np.load(npz)
        info = json.loads(meta.read_text())
        return data["train"], data["val"], data["test"], info["diag"], key, 0.0
    t0 = time.time()
    tr, va, te, diag = orig.fit_fold_mcd(arm_comments, train, val, test, args, seed=args.random_seed + int(fold) * 1000)
    dt = time.time() - t0
    np.savez(npz, train=tr, val=va, test=te)
    meta.write_text(json.dumps({"arm": arm, "fold": fold, "input_hash": input_hash, "seconds": dt,
                                "diag": orig.jsonable(diag)}, indent=1) + "\n")
    events.append({"event": "mcd_fit", "arm": arm, "fold": fold, "key": key, "seconds": dt})
    return tr, va, te, orig.jsonable(diag), key, dt


def fit_select_predict_seeded(x_train, y_train, x_val, y_val, x_test, *, condition: str, trait: str, fold: int,
                              keep_grid_test: bool = False):
    """Original fit_select_predict with an explicit paired SAG random_state."""
    best_score = -1e9
    best_pred = None
    best: dict[str, Any] = {}
    grid: list[dict[str, Any]] = []
    grid_test: dict[str, np.ndarray] = {}
    for feat_n in orig.feature_selection_candidates(int(x_train.shape[1])):
        selector = SelectKBest(f_regression, k=feat_n)
        xtr = sparse.csr_matrix(selector.fit_transform(x_train, y_train))
        xva = sparse.csr_matrix(selector.transform(x_val))
        xte = sparse.csr_matrix(selector.transform(x_test))
        xtr.sort_indices()
        xva.sort_indices()

        def eval_alpha(alpha: float, feat_n=feat_n, xtr=xtr, xva=xva, xte=xte):
            seed = paired_seed(condition, trait, fold, feat_n, alpha)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = Ridge(alpha=alpha, max_iter=20000, solver="sag", random_state=seed)
                model.fit(xtr, y_train)
            conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            val_pred = np.asarray(model.predict(xva), dtype=float)
            score = orig.safe_pearson(y_val, val_pred)
            test_pred = np.asarray(model.predict(xte), dtype=float)
            n_iter = int(np.max(model.n_iter_)) if model.n_iter_ is not None else -1
            return score, alpha, test_pred, n_iter, seed, conv_warn

        results = Parallel(n_jobs=N_JOBS)(delayed(eval_alpha)(alpha) for alpha in ALPHAS)
        for score, alpha, pred, n_iter, seed, conv_warn in results:
            grid.append({"feat_n": str(feat_n), "alpha": float(alpha), "val_pearson": float(score), "n_iter": n_iter,
                         "seed": seed, "convergence_warning": bool(conv_warn)})
            if keep_grid_test:
                grid_test[f"{feat_n}|{float(alpha)}"] = pred
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_pred = pred
                best = {"best_val_pearson": float(score), "best_alpha": float(alpha), "best_feat_n": str(feat_n)}
    if best_pred is None:
        raise RuntimeError("No valid model was selected.")
    return best_pred, best, grid, grid_test


class Context:
    def __init__(self) -> None:
        self.args = original_args()
        self.artifacts = orig.load_baseline_artifacts(BASELINE)
        comments = orig.load_comments(str(COMMENTS), set(self.artifacts["unames"]), self.args.version)
        self.comments = comments
        self.keys = row_keys(comments)
        self.masks, self.deletions = build_arm_masks(comments)
        self.arm_comments = {arm: comments.loc[self.masks[arm]].reset_index(drop=True) for arm in ARMS}
        self.input_hash = {arm: retained_hash(self.keys, self.masks[arm]) for arm in ARMS}
        self.retained = {arm: self.arm_comments[arm]["user_id"].value_counts() for arm in ARMS}
        self.cache_dir = OUT / "mcd_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.frames: dict[tuple[str, str], Any] = {}

    def frame(self, condition: str, trait: str):
        key = (condition, trait)
        if key not in self.frames:
            use_p = bool(orig.CONDITION_SPECS[condition]["use_p"])
            self.frames[key] = orig.trait_frame(self.artifacts, trait, 0, use_p)
        return self.frames[key]


def write_inputs_manifest(ctx: Context) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    venues = sorted({str(s).strip().lower() for s in ctx.comments["subreddit"].fillna("__missing__") if is_trait_named(s)})
    (OUT / "trait_named_venues.txt").write_text("\n".join(venues) + "\n")
    sub = ctx.comments["subreddit"].fillna("__missing__").astype(str).str.strip().str.lower()
    trait = sub.map(is_trait_named)
    venue_counts = (pd.DataFrame({"venue": sub[trait], "user": ctx.comments.loc[trait, "user_id"]})
                    .groupby("venue").agg(n_rows=("user", "size"), n_authors=("user", "nunique"))
                    .sort_values("n_rows", ascending=False).reset_index())
    venue_counts.to_csv(OUT / "deleted_venue_counts.csv", index=False)
    ctx.deletions.to_csv(OUT / "deletion_counts_by_author.csv", index=False)
    manifest = {
        "version": VERSION, "protocol": str(PROTOCOL.relative_to(REPO)), "protocol_sha256": sha256_file(PROTOCOL),
        "protocol_sha256_declared": PROTOCOL_SHA,
        "original_file_sha256": {str(p.relative_to(PERSONA)): sha256_file(p) for p in ORIG_FILES},
        "runner_sha256": sha256_file(Path(__file__)),
        "trait_named_venues_sha256": sha256_file(OUT / "trait_named_venues.txt"),
        "n_trait_named_venues": len(venues),
        "rows": {arm: int(ctx.masks[arm].sum()) for arm in ARMS},
        "input_hash": ctx.input_hash,
        "authors_with_any_trait_row": int((ctx.deletions["n_removed_B"] > 0).sum()),
        "authors_unobserved_BC": int(ctx.deletions["mcd_unobserved_BC"].sum()),
        "trait_rows_also_removed_by_C": int(ctx.deletions["n_trait_rows_removed_by_C"].sum()),
        "original_args": orig.jsonable(vars(ctx.args)),
        "n_jobs": N_JOBS, "blas_threads": 1,
        "versions": {m: __import__(m).__version__ for m in ("numpy", "scipy", "sklearn", "pandas", "joblib")},
    }
    (OUT / "inputs_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if manifest["protocol_sha256"] != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")


def cell_path(condition: str, trait: str, fold: int, arm: str) -> Path:
    return OUT / "cells" / f"{condition}__{trait}__f{fold}__{arm}.json"


def run_cell(ctx: Context, condition: str, trait: str, fold: int, arm: str, keep_grid_test: bool = False) -> dict[str, Any]:
    spec = orig.CONDITION_SPECS[condition]
    frame, _valid, p_matrix = ctx.frame(condition, trait)
    y = frame["label"].to_numpy(float)
    train_idx, val_idx, test_idx = orig.split_indices(frame, fold)
    train_u = frame.iloc[train_idx]["author"].astype(str).tolist()
    val_u = frame.iloc[val_idx]["author"].astype(str).tolist()
    test_u = frame.iloc[test_idx]["author"].astype(str).tolist()
    t0 = time.time()
    tr, va, te, diag, key, mcd_seconds = get_mcd(arm, ctx.arm_comments[arm], ctx.input_hash[arm], train_u, val_u, test_u,
                                                 fold, ctx.args, ctx.cache_dir, ctx.events)
    mcd_full = np.zeros((len(frame), tr.shape[1]), dtype=float)
    mcd_full[train_idx], mcd_full[val_idx], mcd_full[test_idx] = tr, va, te
    dense = [p_matrix, mcd_full] if spec["use_p"] else [mcd_full]
    x_train, x_val, x_test, total = orig.build_design_matrices(ctx.artifacts, frame, train_idx, val_idx, test_idx,
                                                               use_ngram=bool(spec["use_ngram"]), dense_extra=np.hstack(dense))
    t1 = time.time()
    pred, best, grid, grid_test = fit_select_predict_seeded(x_train, y[train_idx], x_val, y[val_idx], x_test,
                                                             condition=condition, trait=trait, fold=fold,
                                                             keep_grid_test=keep_grid_test)
    ridge_seconds = time.time() - t1
    retained = ctx.retained[arm]
    unobserved = [int(retained.get(u, 0) == 0) for u in test_u]
    n_unobs_train = int(sum(retained.get(u, 0) == 0 for u in train_u))
    result = {
        "condition": condition, "trait": trait, "fold": fold, "arm": arm, "version": VERSION,
        "pearson": orig.safe_pearson(y[test_idx], pred), "n_train": len(train_idx), "n_val": len(val_idx),
        "n_test": len(test_idx), "total_features": int(total), **best,
        "mcd_cache_key": key, "mcd_fit_seconds": mcd_seconds, "ridge_seconds": ridge_seconds,
        "cell_seconds": time.time() - t0, "n_unobserved_test": int(sum(unobserved)), "n_unobserved_train": n_unobs_train,
        "n_unobserved_val": int(sum(retained.get(u, 0) == 0 for u in val_u)),
        "grid_n_iter_max": int(max(g["n_iter"] for g in grid)),
        "grid_convergence_warnings": int(sum(g["convergence_warning"] for g in grid)),
        "diag": diag, "grid": grid,
        "predictions": {"author": test_u, "true": [float(v) for v in y[test_idx]], "pred": [float(v) for v in pred],
                        "mcd_unobserved": unobserved},
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    if keep_grid_test:
        result["_grid_test"] = grid_test
    return result


def save_cell(result: dict[str, Any]) -> None:
    path = cell_path(result["condition"], result["trait"], result["fold"], result["arm"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(orig.jsonable(payload)) + "\n")
    tmp.replace(path)


def replay(ctx: Context) -> dict[str, Any]:
    budget = {"stage1_start_epoch": time.time(), "stage1_start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / "budget.json").write_text(json.dumps(budget, indent=1) + "\n")
    result = run_cell(ctx, "np_mcd", "agreeableness", 0, "A", keep_grid_test=True)
    save_cell(result)
    preds = pd.read_csv(ARCH / "predictions.csv")
    arch = preds[(preds.condition_key == "np_mcd") & (preds.trait == "agreeableness") & (preds.fold == 0)]
    metrics = pd.read_csv(ARCH / "metrics_by_fold.csv")
    am = metrics[(metrics.condition_key == "np_mcd") & (metrics.trait == "agreeableness") & (metrics.fold == 0)].iloc[0]
    diags = pd.read_csv(ARCH / "diagnostics.csv")
    ad = diags[(diags.condition_key == "np_mcd") & (diags.trait == "agreeableness") & (diags.fold == 0)].iloc[0]
    mine = result["predictions"]
    gates: dict[str, Any] = {}
    gates["test_authors_equal"] = arch["author"].astype(str).tolist() == mine["author"]
    gates["true_labels_equal"] = bool(np.array_equal(arch["true"].to_numpy(float), np.asarray(mine["true"])))
    for key in EXACT_METRIC:
        gates[f"{key}_equal"] = int(am[key]) == int(result[key])
    for key in EXACT_DIAG:
        gates[f"diag_{key}_equal"] = int(ad[key]) == int(result["diag"][key])
    tau_rel = abs(float(ad["mcd_tau"]) - float(result["diag"]["mcd_tau"])) / abs(float(ad["mcd_tau"]))
    ev_abs = abs(float(ad["comment_svd_explained_variance"]) - float(result["diag"]["comment_svd_explained_variance"]))
    dr = abs(float(am["pearson"]) - float(result["pearson"]))
    a_pred = arch["pred"].to_numpy(float)
    m_pred = np.asarray(mine["pred"], dtype=float)
    pred_corr = float(np.corrcoef(a_pred, m_pred)[0, 1]) if gates["test_authors_equal"] else float("nan")
    gates["mcd_tau_rel_le_1e-3"] = tau_rel <= 1e-3
    gates["svd_ev_abs_le_1e-3"] = ev_abs <= 1e-3
    gates["test_r_abs_diff_le_.02"] = dr <= 0.02
    gates["pred_corr_ge_.98"] = pred_corr >= 0.98
    arch_key = f"{am['best_feat_n']}|{float(am['best_alpha'])}"
    at_arch = result["_grid_test"].get(arch_key)
    reported = {
        "archived": {"pearson": float(am["pearson"]), "best_alpha": float(am["best_alpha"]), "best_feat_n": str(am["best_feat_n"]),
                     "best_val_pearson": float(am["best_val_pearson"]), "mcd_tau": float(ad["mcd_tau"]),
                     "svd_ev": float(ad["comment_svd_explained_variance"]),
                     "min_cluster": int(ad["mcd_min_cluster_size"]), "median_cluster": float(ad["mcd_median_cluster_size"]),
                     "context_ev": float(ad["context_metadata_explained_variance"])},
        "replayed": {"pearson": result["pearson"], "best_alpha": result["best_alpha"], "best_feat_n": result["best_feat_n"],
                     "best_val_pearson": result["best_val_pearson"], "mcd_tau": result["diag"]["mcd_tau"],
                     "svd_ev": result["diag"]["comment_svd_explained_variance"],
                     "min_cluster": result["diag"]["mcd_min_cluster_size"], "median_cluster": result["diag"]["mcd_median_cluster_size"],
                     "context_ev": result["diag"].get("context_metadata_explained_variance")},
        "tau_rel_diff": tau_rel, "svd_ev_abs_diff": ev_abs, "test_r_abs_diff": dr, "pred_corr": pred_corr,
        "hyperparameters_equal": arch_key == f"{result['best_feat_n']}|{float(result['best_alpha'])}",
        "max_abs_pred_diff_over_arch_sd": float(np.max(np.abs(a_pred - m_pred)) / np.std(a_pred)) if gates["test_authors_equal"] else None,
        "diagnostic_only_corr_at_archived_hyperparameters": (float(np.corrcoef(a_pred, at_arch)[0, 1]) if at_arch is not None else None),
        "diagnostic_only_r_at_archived_hyperparameters": (orig.safe_pearson(arch["true"].to_numpy(float), at_arch) if at_arch is not None else None),
    }
    passed = all(bool(v) for v in gates.values())
    elapsed = time.time() - budget["stage1_start_epoch"]
    out = {"stage": "replay", "status": "PASS" if passed else "FAIL_STOP_DIAGNOSE", "gates": {k: bool(v) for k, v in gates.items()},
           "reported": reported, "cell_seconds": result["cell_seconds"], "mcd_fit_seconds": result["mcd_fit_seconds"],
           "ridge_seconds": result["ridge_seconds"], "stage1_elapsed_seconds": elapsed,
           "stage1_within_30min": elapsed <= STAGE1_CAP_S, "grid_n_iter_max": result["grid_n_iter_max"],
           "grid_convergence_warnings": result["grid_convergence_warnings"], "events": ctx.events}
    (OUT / "replay_check.json").write_text(json.dumps(orig.jsonable(out), indent=1) + "\n")
    return out


def cell_plan() -> list[tuple[str, str, int, str]]:
    plan = []
    for condition in CONDITIONS:
        for fold in range(5):
            for trait in orig.BIG5_TRAITS:
                for arm in ARMS:
                    plan.append((condition, trait, fold, arm))
    return plan


def full(ctx: Context) -> dict[str, Any]:
    check = json.loads((OUT / "replay_check.json").read_text())
    variant = VERSION
    if check["status"] != "PASS":
        # Amendment A1: expansion only as a labelled variant after a written label-free diagnosis.
        if not (AMENDMENT.exists() and sha256_file(AMENDMENT) == AMENDMENT_SHA and (OUT / "replay_diagnosis.json").exists()):
            raise RuntimeError("replay gate did not pass and no diagnosed amendment; full stage refused")
        variant = VERSION + "-local"
    start = json.loads((OUT / "budget.json").read_text())["stage1_start_epoch"]
    log = (OUT / "progress.jsonl").open("a")
    timing: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    mcd_times: list[float] = [check["mcd_fit_seconds"]]
    stopped = None
    for condition, trait, fold, arm in cell_plan():
        path = cell_path(condition, trait, fold, arm)
        if path.exists():
            continue
        remaining = BUDGET_S - (time.time() - start)
        ridge_est = max(timing[condition]) if timing[condition] else (check["ridge_seconds"] if condition == "np_mcd" else 60.0)
        mcd_est = max(mcd_times)
        if remaining < 1.1 * (ridge_est + mcd_est):
            stopped = {"reason": "budget", "remaining_seconds": remaining, "next_cell": [condition, trait, fold, arm]}
            break
        try:
            result = run_cell(ctx, condition, trait, fold, arm)
        except Exception as exc:  # recorded, never retried automatically
            rec = {"event": "cell_failed", "cell": [condition, trait, fold, arm], "error": repr(exc), "tb": traceback.format_exc()}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            continue
        save_cell(result)
        timing[condition].append(result["ridge_seconds"])
        if result["mcd_fit_seconds"] > 0:
            mcd_times.append(result["mcd_fit_seconds"])
        log.write(json.dumps({"event": "cell_done", "cell": [condition, trait, fold, arm], "r": result["pearson"],
                              "cell_seconds": result["cell_seconds"], "mcd_fit_seconds": result["mcd_fit_seconds"],
                              "elapsed": time.time() - start, "time": time.strftime("%H:%M:%S")}) + "\n")
        log.flush()
    mask = [{"condition": c, "trait": t, "fold": f, "arm": a, "complete": cell_path(c, t, f, a).exists()} for c, t, f, a in cell_plan()]
    pd.DataFrame(mask).to_csv(OUT / "completion_mask.csv", index=False)
    summary = {"stage": "full", "variant": variant, "replay_gate": check["status"], "stopped": stopped, "complete_cells": int(sum(m["complete"] for m in mask)), "total_cells": len(mask),
               "elapsed_seconds": time.time() - start, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    (OUT / "full_stage_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inputs", "replay", "full"], required=True)
    stage = parser.parse_args().stage
    ctx = Context()
    write_inputs_manifest(ctx)
    if stage == "inputs":
        print(json.dumps(json.loads((OUT / "inputs_manifest.json").read_text()), indent=1))
        return 0
    if stage == "replay":
        if (OUT / "replay_check.json").exists():
            raise RuntimeError("replay already run; not repeated")
        out = replay(ctx)
        print(json.dumps({k: out[k] for k in ("status", "gates", "reported", "cell_seconds", "mcd_fit_seconds", "ridge_seconds", "stage1_elapsed_seconds")}, indent=1, default=str))
        return 0
    print(json.dumps(full(ctx), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

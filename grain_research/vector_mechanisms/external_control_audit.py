"""Independent reconstruction of discussion-control membership and paired effects."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.model_selection import KFold

from .controls import material_hash
from .data_views import stable_key
from .external_heads import task_score


def restrict_oof(result, author_ids):
    """Restrict an old source result to an explicitly identical evaluation cohort."""
    lookup = {u: i for i, u in enumerate(result["ids"])}
    if len(set(author_ids)) != len(author_ids) or not set(author_ids) <= set(lookup):
        raise ValueError("Invalid paired reference author set")
    order = [lookup[u] for u in author_ids]
    return result | {k: result[k][order] for k in ("ids", "y", "p", "folds")}


def paired_seed_average(candidate, references, draws=1000):
    """Bootstrap author differences against mean seed-wise scores, not an ensemble."""
    if not references:
        raise ValueError("Missing comparison runs")
    task = candidate["manifest"]["task"]
    for ref in references:
        if (
            ref["manifest"]["task"] != task
            or ref["manifest"]["permutation_seed"]
            != candidate["manifest"]["permutation_seed"]
            or any(
                not np.array_equal(ref[k], candidate[k]) for k in ("ids", "y", "folds")
            )
        ):
            raise ValueError("Unpaired comparison")
    y, pred = candidate["y"], candidate["p"]
    base = [r["p"] for r in references]
    score = task_score(y, pred, task)
    scores = [task_score(y, p, task) for p in base]
    fold_candidate, fold_reference = [], []
    for fold in range(5):
        selected = candidate["folds"] == fold
        fold_candidate.append(task_score(y[selected], pred[selected], task))
        fold_reference.append(
            np.mean([task_score(y[selected], p[selected], task) for p in base])
        )
    rng, samples = np.random.default_rng(731), []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        samples.append(
            task_score(y[ix], pred[ix], task)
            - np.mean([task_score(y[ix], p[ix], task) for p in base])
        )
    valid = np.asarray(samples)[np.isfinite(samples)]
    if len(valid) < 0.95 * draws:
        raise ValueError("Too many undefined author bootstrap samples")
    return {
        "authors": len(y),
        "candidate_score": score,
        "reference_score": float(np.mean(scores)),
        "reference_seed_min": float(min(scores)),
        "reference_seed_max": float(max(scores)),
        "reference_count": len(base),
        "candidate_mean_fold_score": float(np.mean(fold_candidate)),
        "reference_mean_fold_score": float(np.mean(fold_reference)),
        "mean_fold_delta": float(np.mean(np.subtract(fold_candidate, fold_reference))),
        "positive_fold_count": int(
            np.sum(np.subtract(fold_candidate, fold_reference) > 0)
        ),
        "delta": float(score - np.mean(scores)),
        "lower": float(np.quantile(valid, 0.025)),
        "upper": float(np.quantile(valid, 0.975)),
        "draws": draws,
        "undefined_draws": draws - len(valid),
        "scope": "paired authors conditional on fitted OOF; seed scores averaged, not a blended head",
    }


def independent_control_records(bundles, bridge, task, mode, seed):
    """Reconstruct split membership without calling the fitted runner's selectors."""
    ids = np.asarray([b["author_id"] for b in bundles])
    if len(set(ids)) != len(ids) or set(ids) & {b["author_id"] for b in bridge}:
        raise ValueError("Duplicate or overlapping source authors")
    folds = np.asarray([b["official_folds"][task] for b in bundles])
    threads = {
        b["author_id"]: {c["thread_id"] for v in b["views"] for c in v}
        for b in bundles + bridge
    }
    if any(not t or "" in t for t in threads.values()):
        raise ValueError("Missing discussion provenance")
    records = []

    def select(train, heldout, scope, target=None):
        train_ids = [b["author_id"] for b in train]
        test_ids = [b["author_id"] for b in heldout]
        held = set().union(*(threads[u] for u in test_ids))
        shared = np.array([bool(threads[u] & held) for u in train_ids])
        clean = np.flatnonzero(~shared)
        if mode == "retain":
            kept = np.arange(len(train))
        elif mode == "purge":
            kept = clean
        elif mode == "random_count":
            count = len(clean) if target is None else target
            ranking = sorted(
                range(len(train)),
                key=lambda i: stable_key(train_ids[i], f"vm05_drop:{seed}:{scope}"),
            )
            kept = np.sort(ranking[:count])
        else:
            raise ValueError("Unknown control")
        records.append(
            {
                "scope": scope,
                "mode": mode,
                "seed": seed,
                "train_available": len(train),
                "test_authors": len(heldout),
                "shared_thread_authors": int(shared.sum()),
                "train_retained": len(kept),
                "retained_shared_thread_authors": int(shared[kept].sum()),
                "train_author_commitment": material_hash([train_ids[i] for i in kept]),
                "test_author_commitment": material_hash(test_ids),
                "label_values_used_for_selection": False,
                "matched_reference_count": target,
            }
        )
        return [train[i] for i in kept], [train[i] for i in clean]

    outer_pairs = [
        (
            [b for b, f in zip(bundles, folds) if f != fold],
            [b for b, f in zip(bundles, folds) if f == fold],
            f"outer:{fold}",
        )
        for fold in range(5)
    ] + [(bundles, bridge, "bridge_source_fit")]
    for train, heldout, scope in outer_pairs:
        kept, purged = select(train, heldout, scope)
        for ti, vi in KFold(3, shuffle=True, random_state=731).split(kept):
            target = None
            if mode == "random_count":
                held = set().union(*(threads[purged[i]["author_id"]] for i in vi))
                target = sum(not threads[purged[i]["author_id"]] & held for i in ti)
            select(
                [kept[i] for i in ti],
                [kept[i] for i in vi],
                f"{scope}:inner:{material_hash(vi.tolist())[:12]}",
                target,
            )
    return records


def audit_discussion_controls(folder, result, bundles, bridge):
    """Verify all outer/inner/full-source selection records against frozen material."""
    m, s = result["manifest"], result["summary"]
    if material_hash(bundles) != m["source_material_sha256"]:
        raise ValueError("Different frozen source evidence")
    if set(result["ids"]) != {b["author_id"] for b in bundles}:
        raise ValueError("Missing source predictions")
    raw = json.loads((folder / "discussion_control.json").read_text())
    expected = independent_control_records(
        bundles, bridge, m["task"], m["thread_control"], m["drop_seed"]
    )
    if raw != expected or len(raw) != 24:
        raise ValueError(
            "Stored discussion control cannot be independently reconstructed"
        )
    selected = json.loads((folder / "selection.json").read_text())
    outer = [r for r in expected if ":inner:" not in r["scope"]]
    if len(selected) != 5 or any(
        r["fold"] != i
        or r["train_n"] != outer[i]["train_retained"]
        or r["test_n"] != outer[i]["test_authors"]
        for i, r in enumerate(selected)
    ):
        raise ValueError("Fitted fold sizes disagree with reconstructed selection")
    if (
        s["full_train_authors"] != outer[-1]["train_retained"]
        or hashlib.sha256(
            (folder / "private_full_source_head.joblib").read_bytes()
        ).hexdigest()
        != s["full_head_sha256"]
        or m["bridge_labels_used"]
        or s["bridge_labels_used"]
    ):
        raise ValueError("Full-source head seal or label boundary differs")
    return {
        "case": folder.name,
        "selection_records": len(expected),
        "outer_min_train": min(r["train_retained"] for r in outer[:5]),
        "outer_max_train": max(r["train_retained"] for r in outer[:5]),
        "inner_min_train": min(
            r["train_retained"] for r in expected if ":inner:" in r["scope"]
        ),
        "full_source_train": outer[-1]["train_retained"],
        "max_retained_shared_authors": max(
            r["retained_shared_thread_authors"] for r in expected
        ),
        "membership_reconstruction": "PASS",
    }, expected


def audit_count_match(purge_records, random_records):
    """Require matching training counts at all 24 selections, not only outer folds."""
    if len(purge_records) != 24 or len(random_records) != 24:
        raise ValueError("Incomplete nested count controls")
    fields = ("scope", "train_available", "train_retained", "test_authors")
    if any(
        a["mode"] != "purge"
        or b["mode"] != "random_count"
        or any(a[k] != b[k] for k in fields)
        for a, b in zip(purge_records, random_records)
    ):
        raise ValueError("Nested purge/random training counts do not match")
    return 24

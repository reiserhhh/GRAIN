#!/usr/bin/env python3
"""R31 shortcut check: topic composition vs historical readouts; task/wording vs donor action."""

from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R31_SHORTCUT_PROTOCOL_V1.md"
PROTOCOL_SHA = "6c81124072eb184965de483333e6fb8a607df8e36a3e619919718e9d52729770"
LB = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2/pandora"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F10/shortcut_r1"
SEED, PERM = 20260926, 500
TRAITS = ("Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def loo_r2(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    pred = np.empty_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        pred[i] = X[i] @ np.linalg.lstsq(X[m], y[m], rcond=None)[0]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def anova_share(y, groups):
    g = pd.Series(y).groupby(groups)
    between = (g.transform("mean") - y.mean()) ** 2
    return float(between.sum() / ((y - y.mean()) ** 2).sum())


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.prepare_grain_predecessor_reference import DEFAULT

    material = json.loads((DEFAULT / "private_material.json").read_text())
    subs = [[c["subreddit"] for v in b["views"] for c in v] for b in material["bundles"]]
    with np.load(LB / "private_predictions.npz", allow_pickle=False) as z:
        held = z["held_indices"]
        preds = {arm: z[arm] for arm in ("semantic", "grain")}
    counts = collections.Counter(s for i in held for s in subs[int(i)])
    top = [s for s, _ in counts.most_common(30)]
    share = np.array([[sum(x == s for x in subs[int(i)]) / len(subs[int(i)]) for s in top]
                      + [sum(x not in top for x in subs[int(i)]) / len(subs[int(i)])] for i in held])
    centered = share - share.mean(0)
    u, sv, _ = np.linalg.svd(centered, full_matrices=False)
    pcs = u[:, :5] * sv[:5]
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(held)) for _ in range(PERM)]
    part_a = {}
    for arm, p in preds.items():
        for t, trait in enumerate(TRAITS):
            y = p[:, t]
            r2 = loo_r2(pcs, y)
            null = [loo_r2(pcs[pi], y) for pi in perms]
            part_a[f"{arm}|{trait}"] = {"cv_r2": float(r2), "null_q95": float(np.percentile(null, 95)),
                                        "flagged": bool(r2 > np.percentile(null, 95))}
    eff = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    part_b = {}
    for q, g in eff[eff.kind == "donor"].groupby("query"):
        y = g.effect.to_numpy()
        for factor in ("situation", "family"):
            share_v = anova_share(y, g[factor].to_numpy())
            lab = g[factor].to_numpy()
            null = [anova_share(y, lab[rng.permutation(len(lab))]) for _ in range(PERM)]
            part_b[f"donor_action|{q}|{factor}"] = {"share": share_v, "null_q95": float(np.percentile(null, 95)),
                                                    "levels": int(g[factor].nunique()),
                                                    "flagged": bool(share_v > np.percentile(null, 95))}
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "held_authors": len(held),
           "top_subreddits": len(top), "topic_share_matrix_other_mean": float(share[:, -1].mean()),
           "part_a_topic_vs_readout": part_a, "part_b_task_wording_vs_action": part_b,
           "neural_calls": 0, "labels_read": False, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "shortcuts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("part_a_topic_vs_readout", "part_b_task_wording_vs_action", "topic_share_matrix_other_mean")}, indent=1))


if __name__ == "__main__":
    main()

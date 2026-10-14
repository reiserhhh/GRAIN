#!/usr/bin/env python3
"""Post-result independent formulas for M07 scores, effects and order contrasts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    """Read a local metadata artifact."""
    return json.loads(path.read_text())


def sha(path):
    """Hash retained bytes without altering the packet."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path):
    """Load arrays without importing the production scoring routines."""
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def main(out):
    """Recompute scalar/state identities and descriptive binding-by-order effects."""
    root = Path(__file__).resolve().parents[1]
    m = read(out / "manifest.json")
    for section in ("code", "source"):
        for n, h in m[section].items():
            assert sha(root / n) == h, n
    for n, h in m["inputs"].items():
        assert sha(out / n) == h, n
    bm = read(out / "bundle/manifest.json")
    assert sha(out / "bundle/manifest.json") == m["bundle_manifest_sha256"]
    for n, h in bm["files"].items():
        assert sha(out / "bundle" / n) == h, n
    for folder, name in (
        (out, "analysis_freeze.json"),
        (out / "cold_context", "freeze.json"),
        (out / "cold_token", "freeze.json"),
    ):
        for n, h in read(folder / name).items():
            assert sha(folder / n) == h, n
    refs = load(out / "bundle/reader.npz")
    models = {q: load(out / f"bundle/state_{q}.npz") for q in m["alphabets"]}
    preds = {r["id"]: r for r in read(out / "cold_context/predictions.json")}
    design = {d["id"]: d for d in read(out / "design.json")}
    worst = {
        "native": 0.0,
        "teacher_affine": 0.0,
        "reconstructed_state": 0.0,
        "reconstructed_score": 0.0,
        "carrier_sum": 0.0,
        "effect": 0.0,
    }
    checks = 0
    for i, d in design.items():
        q, r = d["query"], preds[i]
        tm = read(out / f"teacher/input_{int(i):03d}.json")
        tp = out / f"teacher/input_{int(i):03d}.npz"
        assert sha(tp) == tm["array_sha256"]
        logits = np.asarray(tm["native"]["candidate_logits"])
        weights = np.exp(logits - logits.max())
        actual = weights @ np.asarray(m["alphabets"][q]["values"]) / weights.sum()
        worst["native"] = max(
            worst["native"], abs(actual - tm["native"]["conditional_score"])
        )
        z = load(out / f"cold_context/input_{int(i):03d}.npz")
        token = load(out / f"cold_token/input_{int(i):03d}.npz")
        for a in (z, token):
            np.testing.assert_array_equal(a["ids"], d["ids"])
            np.testing.assert_array_equal(a["mask"], d["mask"])
        h = load(tp)["state"]
        mask = np.asarray(d["mask"], bool)
        zp, zq, g, gf, anchor = (refs[q + "__" + n] for n in ("Z", "h", "G", "g", "b"))

        def direct(vector, zp=zp, zq=zq, g=g, gf=gf, anchor=anchor):
            """Contract the original reader arrays without the production class."""
            width = len(zq)
            return float(
                anchor
                + np.sum((vector[:-width].reshape(zp.shape) - zp) * g)
                + (vector[-width:] - zq) @ gf
            )

        worst["teacher_affine"] = max(
            worst["teacher_affine"],
            abs(direct(np.r_[h[mask].ravel(), h[-1]]) - tm["affine"]),
        )
        x = np.r_[z["hidden"][mask].ravel(), z["hidden"][-1]].astype(float)
        mod = models[q]
        k = ((x - mod["mean_x"]) / mod["scale"]) @ mod["train_x"].T
        estimated = (
            x
            + mod["mean_r"]
            + (k - k.mean() - mod["col1"] + mod["grand1"]) @ mod["coef1"]
        )
        worst["reconstructed_state"] = max(
            worst["reconstructed_state"],
            float(abs(estimated - z["upper_estimated"]).max()),
        )
        worst["reconstructed_score"] = max(
            worst["reconstructed_score"], abs(direct(estimated) - r["A_state"])
        )
        worst["carrier_sum"] = max(
            worst["carrier_sum"],
            abs(
                r["context_all"]
                - r["context_C"]
                - r["context_PQ"]
                + r["context_origin"]
            ),
        )
        checks += 9
    f = pd.read_csv(out / "levels.csv").set_index("id")
    effects = pd.read_csv(out / "effects.csv")
    methods = [
        "native",
        "upper_affine",
        "token_uniform",
        "token_semantic_order",
        "A_state",
        "A_compiled",
        "context_all",
        "context_C",
        "context_PQ",
        "B_C",
        "B_P",
        "B_Q",
        "A_P",
        "A_Q",
    ]
    for _, r in effects.iterrows():
        for n in methods:
            err = abs(f.loc[r.donor, n] - f.loc[r.base, n] - r[n])
            worst["effect"] = max(worst["effect"], err)
            checks += 1
    assert max(worst.values()) < 1e-8, worst
    rows = []
    for (frame, query), group in f[f.family == "binding"].groupby(["frame", "query"]):
        grid = group.set_index(["self_high", "order"])
        for method in methods[:9]:
            d0 = grid.loc[(1, 0), method] - grid.loc[(0, 0), method]
            d1 = grid.loc[(1, 1), method] - grid.loc[(0, 1), method]
            rows.append(
                {
                    "frame": frame,
                    "query": query,
                    "method": method,
                    "own_target": query == group.target.iloc[0],
                    "order0_effect": d0,
                    "order1_effect": d1,
                    "binding_main": (d0 + d1) / 2,
                    "binding_order_interaction": (d0 - d1) / 2,
                }
            )
    pd.DataFrame(rows).to_csv(out / "binding_order_components.csv", index=False)
    binding = effects[effects.kind == "binding"]
    bounds = []
    for label, g in (
        ("all_queries", binding),
        ("own_target", binding[binding["query"] == binding.target]),
    ):
        assert g.bag_equal.all()
        bounds.append(
            {
                "subset": label,
                "pairs": len(g),
                "any_bag_function_native_rmse_lower_bound": float(
                    np.sqrt(np.mean(g.native**2)) / 2
                ),
                "uniform_max_abs_binding_delta": float(abs(g.token_uniform).max()),
            }
        )
    result = {
        "status": "PASS",
        "formula_and_input_checks": checks,
        "max_errors": worst,
        "bag_bound": bounds,
        "new_model_calls": 0,
        "independent_researcher": False,
        "post_result_analysis": "binding/order factorial and pairwise bag lower bound; no fitting",
        "auditor_sha256": sha(Path(__file__)),
        "binding_table_sha256": sha(out / "binding_order_components.csv"),
    }
    (out / "independent_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)

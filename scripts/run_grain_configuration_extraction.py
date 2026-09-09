#!/usr/bin/env python3
"""NEXT05: fixed lightweight programs for configuration, with author-held testing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.configuration_extraction import (
    ARMS,
    InputProgram,
    token_features,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import question
from grain_research.vector_mechanisms.external_material import unpack_features
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.natural_transport import (
    fit_operator,
    paired_summary,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_conditional_edits import load_panel
from scripts.run_grain_natural_transport import FACTORS, QUERIES, SOURCE

PLAN = ROOT / "docs/GRAIN_LOGIC_CONFIGURATION_EXTRACTION_PLAN_V1.md"
TRANSPORT = BASE / "logic_natural_transport_v1"
RESIDUAL = BASE / "logic_query_residual_v1"
S1 = BASE / "multiquery_sensitivity_s1_r1"
SOURCE_TEXT = BASE / "source_remainder_s9_r1/private_material.json"
TABLE = BASE / "navigation_v1/NAV-03/input_pullback_r1/embedding_table.npy"


def load_features(out):
    """Bind source/held bag vectors to exact cached text, without criterion reads."""
    from transformers import AutoTokenizer

    source_factors = pd.read_csv(
        BASE / "factor_path_routes_v1/source_r2/private_factor_views.csv"
    )
    paths = pd.read_csv(BASE / "dynamic_path_channels_v1/private_path_views.csv")
    held = pd.read_csv(TRANSPORT / "private_factor_scores.csv")
    source_ids = set(source_factors.author_id)
    held_ids = set(held.author_id)
    if len(source_ids) != 276 or len(held_ids) != 12 or source_ids & held_ids:
        raise ValueError("Wrong or overlapping author roles")
    sm = read(SOURCE_TEXT.parent / "manifest.json")
    if sha(SOURCE_TEXT) != sm["material_sha256"]:
        raise ValueError("Source material changed")
    texts = {
        r["author_id"]: r for r in read(SOURCE_TEXT) if r["author_id"] in source_ids
    }
    cache_index = (
        BASE / "response_student_source_connection_r2/private_cache_hashes.json"
    )
    cache = {Path(k).stem: (ROOT / k, v) for k, v in read(cache_index).items()}
    e5tok = AutoTokenizer.from_pretrained(
        "intfloat/e5-large-v2", local_files_only=True, trust_remote_code=False
    )
    modeldir = read(S1 / "manifest.json")["model_dir"]
    qtok = AutoTokenizer.from_pretrained(
        modeldir, local_files_only=True, trust_remote_code=False
    )
    table = np.load(TABLE, mmap_mode="r", allow_pickle=False)
    hashes = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            PLAN,
            Path(__file__),
            ROOT / "grain_research/vector_mechanisms/configuration_extraction.py",
            SOURCE_TEXT,
            cache_index,
            TABLE,
        )
    }
    records, bags, tokens = [], [], []
    signatures = 0
    identities = set()

    def append(uid, view, edit, group, bodies, vectors, factor):
        """Store numeric observations only; raw strings never enter report rows."""
        records.append(
            dict(
                author_id=uid,
                view=view,
                edit=edit,
                group=group,
                **dict(zip(FACTORS, factor, strict=True)),
            )
        )
        bags.append(np.asarray(vectors, float))
        ids = [
            i for body in bodies for i in qtok.encode(body, add_special_tokens=False)
        ]
        tokens.append(
            np.stack(
                [
                    token_features(
                        ids, table, qtok.encode(question(q), add_special_tokens=False)
                    )
                    for q in QUERIES
                ]
            )
        )

    factor_lookup = source_factors.set_index(["author_id", "view"])
    for n, uid in enumerate(sorted(source_ids)):
        path, expected = cache[uid]
        meta = read(path.with_suffix(".json"))
        if (
            sha(path) != expected
            or expected != meta["feature_sha256"]
            or meta["author_id"] != uid
        ):
            raise ValueError("Source encoder cache identity changed")
        manifest = read(path.parent.parent / "manifest.json")
        identities.add(manifest["identity"])
        if (
            manifest["model"]["model"] != "intfloat/e5-large-v2"
            or manifest["model"]["prefix"] != "query: "
        ):
            raise ValueError("Unexpected encoder or prefix")
        raw = [t for v in texts[uid]["views"] for t in v["texts"]]
        signature = material_hash(
            e5tok(["query: " + t for t in raw], truncation=False)["input_ids"]
        )
        if signature != meta["token_signatures"]["cue_mask"]:
            raise ValueError("E5 input does not match fixed query text")
        signatures += 1
        hashes[str(path.relative_to(ROOT))] = expected
        hashes[str(path.with_suffix(".json").relative_to(ROOT))] = sha(
            path.with_suffix(".json")
        )
        with np.load(path, allow_pickle=False) as z:
            vectors = unpack_features(dict(z), meta["feature_aliases"])[
                "cue_mask_comment_vectors"
            ]
        if vectors.shape != (3, 12, 1024):
            raise ValueError("Incomplete source view bags")
        for view in range(3):
            append(
                uid,
                view,
                "source",
                texts[uid]["source_group"],
                texts[uid]["views"][view]["texts"],
                vectors[view],
                factor_lookup.loc[(uid, view), FACTORS].to_numpy(float),
            )
        if n % 80 == 0:
            print(f"Prepared source author {n + 1}/276", flush=True)
    s1m = read(S1 / "manifest.json")
    if (
        sha(S1 / "private_encoder.npz") != s1m["encoder_sha256"]
        or sha(S1 / "private_material.json") != s1m["material_sha256"]
    ):
        raise ValueError("Held text/encoder changed")
    hashes[str((S1 / "private_encoder.npz").relative_to(ROOT))] = s1m["encoder_sha256"]
    hashes[str((S1 / "private_material.json").relative_to(ROOT))] = s1m[
        "material_sha256"
    ]
    held_lookup = held.set_index(["author_id", "edit"])
    with np.load(S1 / "private_encoder.npz", allow_pickle=False) as z:
        hbags, poles = z["comment_vectors"], z["poles"]
    for index, row in enumerate(read(S1 / "private_material.json")):
        uid = row["author_id"]
        if uid not in held_ids or row["view"] != 0:
            continue
        for edit in ("full", "e1", "e2"):
            keep = [j for j in range(12) if edit == "full" or j not in row[edit]]
            append(
                uid,
                0,
                edit,
                "held",
                [row["texts"][j] for j in keep],
                hbags[index, keep],
                held_lookup.loc[(uid, edit), FACTORS].to_numpy(float),
            )
    meta = pd.DataFrame(records)
    if len(meta) != 864 or meta.duplicated(["author_id", "view", "edit"]).any():
        raise ValueError("Incomplete fixed observation panel")
    token = np.asarray(tokens)
    offsets = np.cumsum([0] + [len(b) for b in bags])
    np.savez_compressed(
        out / "private_features.npz",
        bags=np.concatenate(bags),
        offsets=offsets,
        token=token,
        poles=poles,
    )
    meta.to_csv(out / "private_feature_rows.csv", index=False)
    write(
        out / "feature_manifest.json",
        {
            "hashes": hashes,
            "source_token_signatures": signatures,
            "observations": len(meta),
            "new_contextual_forwards": 0,
            "encoder_identities": sorted(identities),
            "e5_prefix": "query: ",
            "token_lookup_layers": 0,
            "B4": "not collected: existing designs differ in query/material",
        },
    )
    return meta, bags, token, poles, paths


def targets_for(meta, paths, query):
    """Join one frozen function component, never external author labels."""
    group = "mbti_only" if query == "TF_cont" else "big5_only"
    part = meta[meta.group == group].sort_values(["author_id", "view"]).copy()
    part["feature_index"] = part.index
    joined = part.merge(
        paths[paths["query"] == query][["author_id", "view", "configuration"]],
        on=["author_id", "view"],
        validate="one_to_one",
    )
    if (
        joined.author_id.nunique() != (63 if query == "TF_cont" else 213)
        or len(joined) % 3
    ):
        raise ValueError("Incorrect source target support")
    return joined


def fit_and_predict(out, meta, bags, tokens, poles, paths):
    """Fit reference maps within each author fold and freeze all held predictions."""
    oof, held_rows = [], []
    prior_maps = read(RESIDUAL / "part_operators.json")
    for qi, q in enumerate(QUERIES):
        data = targets_for(meta, paths, q)
        ids = data.author_id.unique()
        xg = data[FACTORS].to_numpy().reshape(-1, 3, len(FACTORS))
        yc = data.configuration.to_numpy().reshape(-1, 3)
        splits = list(KFold(5, shuffle=True, random_state=20260910).split(ids))
        splits.append((np.arange(len(ids)), np.array([], int)))
        for fold, (train, test) in enumerate(splits):
            final = fold == 5
            gm = fit_operator(xg[train], yc[train], "between")
            if (
                final
                and np.max(
                    np.abs(
                        np.asarray(gm["coef"]) - prior_maps[f"{q}::between"][1]["coef"]
                    )
                )
                > 1e-10
            ):
                raise ValueError(
                    "Final reference differs from prior configuration score"
                )
            tr = np.concatenate([np.arange(i * 3, (i + 1) * 3) for i in train])
            train_data = data.iloc[tr]
            gx = train_data[FACTORS].to_numpy() @ gm["coef"] + gm["intercept"]
            y = np.column_stack(
                [train_data.configuration, train_data.configuration - gx]
            )
            ix = train_data.feature_index.to_numpy(int)
            train_bags = [bags[i] for i in ix]
            if final:
                test_data = meta[meta.group == "held"].copy()
                tx = test_data.index.to_numpy(int)
            else:
                te = np.concatenate([np.arange(i * 3, (i + 1) * 3) for i in test])
                test_data = data.iloc[te]
                tx = test_data.feature_index.to_numpy(int)
            test_g = test_data[FACTORS].to_numpy() @ gm["coef"] + gm["intercept"]
            for arm in ARMS:
                model = InputProgram(arm, poles[qi]).fit(train_bags, tokens[ix, qi], y)
                prediction = model.predict([bags[i] for i in tx], tokens[tx, qi])
                solo = model.predict([bags[tx[0]]], tokens[tx[:1], qi])[0]
                if np.max(np.abs(solo - prediction[0])) > 1e-10:
                    raise ValueError("Scoring depends on test batch")
                for i, (_, r) in enumerate(test_data.iterrows()):
                    row = {
                        "author_id": r.author_id,
                        "view": int(r.view),
                        "edit": r.edit,
                        "query": q,
                        "arm": arm,
                        "fold": fold,
                        "predicted_c": prediction[i, 0],
                        "predicted_r": prediction[i, 1],
                        "derived_r": prediction[i, 0] - test_g[i],
                        "grain_part": test_g[i],
                    }
                    if final:
                        held_rows.append(row)
                    else:
                        row.update(
                            actual_c=float(r.configuration),
                            actual_r=float(r.configuration - test_g[i]),
                        )
                        oof.append(row)
                if final:
                    joblib.dump(
                        {
                            "input": model,
                            "grain": gm,
                            "query": q,
                            "target_order": ["configuration", "configuration_residual"],
                        },
                        out / f"private_{q}_{arm}.joblib",
                    )
            print(
                f"Fit {q} {'final' if final else 'author fold ' + str(fold)}",
                flush=True,
            )
    pd.DataFrame(oof).to_csv(out / "private_source_oof.csv", index=False)
    pd.DataFrame(held_rows).to_csv(out / "private_held_predictions.csv", index=False)
    write(
        out / "prediction_freeze.json",
        {
            "source_sha256": sha(out / "private_source_oof.csv"),
            "held_sha256": sha(out / "private_held_predictions.csv"),
            "native_held_targets_read": False,
            "human_labels_read": False,
            "new_neural_calls": 0,
        },
    )


def evaluate(out):
    """Only after predictions freeze, join cached independent model endpoints."""
    freeze = read(out / "prediction_freeze.json")
    if sha(out / "private_held_predictions.csv") != freeze["held_sha256"]:
        raise ValueError("Predictions changed")
    oof = pd.read_csv(out / "private_source_oof.csv")
    pred = pd.read_csv(out / "private_held_predictions.csv")
    rows = []
    for (q, arm), g in oof.groupby(["query", "arm"]):
        for target, estimate in (
            ("actual_c", "predicted_c"),
            ("actual_r", "predicted_r"),
            ("actual_r", "derived_r"),
        ):
            for level in ("author", "within"):
                means = g.groupby("author_id")[[target, estimate]].transform("mean")
                if level == "author":
                    panel = g.groupby("author_id")[[target, estimate]].mean()
                    stats = paired_summary(panel[target], panel[estimate])
                else:
                    # Report within correlation descriptively; no view-IID interval.
                    from grain_research.vector_mechanisms.natural_transport import (
                        correlation,
                    )

                    stats = {
                        "n": g.author_id.nunique(),
                        "r": correlation(
                            g[target] - means[target], g[estimate] - means[estimate]
                        ),
                    }
                rows.append(
                    dict(
                        query=q,
                        arm=arm,
                        split="source_oof",
                        level=level,
                        target=target,
                        estimate=estimate,
                        **stats,
                    )
                )
    native, states, _, _ = load_panel()
    sm = read(SOURCE / "manifest.json")
    with np.load(
        Path(sm["functions"]) / "nine_query_functions.npz", allow_pickle=False
    ) as z:
        funcs = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in QUERIES
        }
    native_edits = pd.read_csv(TRANSPORT / "private_predictions.csv")
    targets = []
    for i, n in native.iterrows():
        f = funcs[n["query"]]
        targets.append(
            {
                "author_id": n.author_id,
                "query": n["query"],
                "c_full": f.parts(states[i]["full_P"], states[i]["full_Q"])[
                    "configuration"
                ],
                "c_e1": f.parts(states[i]["edited_P"], states[i]["edited_Q"])[
                    "configuration"
                ],
            }
        )
    target = pd.DataFrame(targets).set_index(["author_id", "query"])
    effects = []
    for (q, arm), g in pred.groupby(["query", "arm"]):
        full = g[g.edit == "full"].set_index("author_id").sort_index()
        t = target.xs(q, level="query").loc[full.index]
        for col in ("predicted_c", "predicted_r", "derived_r"):
            actual = t.c_full.to_numpy() - (
                full.grain_part.to_numpy() if col != "predicted_c" else 0
            )
            rows.append(
                dict(
                    query=q,
                    arm=arm,
                    split="held",
                    level="full",
                    target="c" if col == "predicted_c" else "r",
                    estimate=col,
                    **paired_summary(actual, full[col]),
                )
            )
        for edit in ("e1", "e2"):
            changed = g[g.edit == edit].set_index("author_id").loc[full.index]
            native_rows = (
                native_edits[(native_edits["query"] == q) & (native_edits.edit == edit)]
                .set_index("author_id")
                .loc[full.index]
            )
            for col in ("predicted_c", "predicted_r", "derived_r"):
                d = (changed[col] - full[col]).to_numpy()
                # Components are not complete native-score predictions; r is the primary relation.
                rows.append(
                    dict(
                        query=q,
                        arm=arm,
                        split="held",
                        level=edit,
                        target="native_delta",
                        estimate=col,
                        **paired_summary(native_rows.native_delta, d, calibrated=False),
                    )
                )
                if edit == "e1":
                    actual = (t.c_e1 - t.c_full).to_numpy() - (
                        changed.grain_part - full.grain_part
                    ).to_numpy() * (col != "predicted_c")
                    rows.append(
                        dict(
                            query=q,
                            arm=arm,
                            split="held",
                            level=edit,
                            target="c_delta" if col == "predicted_c" else "r_delta",
                            estimate=col,
                            **paired_summary(actual, d),
                        )
                    )
                effects.extend(
                    {
                        "author_id": uid,
                        "query": q,
                        "arm": arm,
                        "edit": edit,
                        "estimate": col,
                        "delta": v,
                        "native_delta": n,
                    }
                    for uid, v, n in zip(
                        full.index, d, native_rows.native_delta, strict=True
                    )
                )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "private_held_changes.csv", index=False)
    print(
        pd.DataFrame(rows)
        .query("split=='held' and estimate=='predicted_c'")[
            ["query", "arm", "level", "target", "r"]
        ]
        .to_string(index=False),
        flush=True,
    )


def main():
    """One bounded numerical comparison; no retry/tuning loop based on outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_configuration_extraction_v1"
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Use private results directory")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write(out / "status.json", {"status": "RUNNING"})
    try:
        with threadpool_limits(limits=2):
            values = load_features(out)
            fit_and_predict(out, *values)
            evaluate(out)
        write(
            out / "status.json",
            {"status": "COMPLETE", "seconds": time.monotonic() - start},
        )
    except Exception as exc:
        write(
            out / "status.json", {"status": "FAILED", "error_type": type(exc).__name__}
        )
        raise


if __name__ == "__main__":
    main()

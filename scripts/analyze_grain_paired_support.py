#!/usr/bin/env python3
"""Fit the frozen support factorial, then publish matched aggregate evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.factor_path_routes import paired_delta
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.natural_transport import (
    correlation,
    paired_summary,
)
from grain_research.vector_mechanisms.paired_support import fit_support, predict_support
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.run_grain_paired_support import OUT, QUERIES, verify

TARGETS = ("configuration", "score")


def construct(out):
    """Fit only fixed model functions, keeping complete authors within folds."""
    manifest, status = verify(out), read(out / "status.json")
    if (
        status["status"] != "COMPLETE"
        or status["completed"] != 88
        or status["calls"] != 180
    ):
        raise ValueError("Declared full panel required")
    design = read(out / "private_design.json")
    with np.load(out / "private_e5_inputs.npz", allow_pickle=False) as z:
        e5 = [z["mean"], z["ordinal"]]
    with np.load(manifest["functions"], allow_pickle=False) as z:
        fs = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in QUERIES
        }
    rows, errors = [], []
    for q in QUERIES:
        selected = [d for d in design if d["query"] == q]
        upper, native, p4, q4 = [], [], [], []
        for d in selected:
            path = out / "private_packets" / f"pair_{d['index']:03d}.json"
            packet = read(path)
            if sha(path.with_suffix(".npz")) != packet["arrays_sha256"]:
                raise ValueError("Paired packet changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                pp, qq = [], []
                for order in ("original", "reverse"):
                    actual = fs[q].parts(z[order + "_P"], z[order + "_Q"])
                    errors.extend(abs(v - packet[order][k]) for k, v in actual.items())
                    pp.append(z[order + "_earlyP"].reshape(-1).astype(float))
                    qq.append(z[order + "_earlyQ"].astype(float))
                p4.append(pp)
                q4.append(qq)
            upper.append(
                [[packet[o][k] for k in TARGETS] for o in ("original", "reverse")]
            )
            native.append([packet[o]["native"] for o in ("original", "reverse")])
        y, native = np.asarray(upper), np.asarray(native)
        source = np.array([i for i, d in enumerate(selected) if d["role"] == "source"])
        held = np.array([i for i, d in enumerate(selected) if d["role"] == "transport"])
        if (len(source), len(held)) != (32, 12):
            raise ValueError("Wrong paired author populations")
        idx = [d["index"] for d in selected]
        variants = {
            "ordinal": [x[idx] for x in e5],
            "early_pq": [np.array(p4), np.array(q4)],
        }
        folds = [
            (source[tr], source[te])
            for tr, te in KFold(4, shuffle=True, random_state=20260910).split(source)
        ]
        folds.append((source, held))
        for name, blocks in variants.items():
            for support in ("natural", "paired"):
                for fold, (train, test) in enumerate(folds):
                    model = fit_support(blocks, y, train, support == "paired")
                    pred = predict_support(model, [x[test] for x in blocks])
                    if fold == 4:
                        joblib.dump(model, out / f"private_{q}_{name}_{support}.joblib")
                    for i, j in enumerate(test):
                        d = selected[j]
                        for o, order in enumerate(("original", "reverse")):
                            rows.append(
                                {
                                    "author_id": d["author_id"],
                                    "query": q,
                                    "role": d["role"],
                                    "input": name,
                                    "support": support,
                                    "fold": fold,
                                    "order": order,
                                    "actual_c": y[j, o, 0],
                                    "actual_f": y[j, o, 1],
                                    "native": native[j, o],
                                    "predicted_c": pred[i, o, 0],
                                    "predicted_f": pred[i, o, 1],
                                }
                            )
    if max(errors) > 1e-10:
        raise ValueError("Independent fixed-function replay failed")
    predictions = pd.DataFrame(rows)
    predictions.to_csv(out / "private_predictions.csv", index=False)
    write(
        out / "fit_freeze.json",
        {
            "plan_sha256": manifest["code_hashes"][
                "docs/GRAIN_LOGIC_CONDITIONAL_SUPPORT_PLAN_V1.md"
            ],
            "predictions_sha256": sha(out / "private_predictions.csv"),
            "script_sha256": sha(Path(__file__)),
            "pq_replay_max_error": max(errors),
            "human_criteria_read": False,
            "model_hashes": {
                p.name: sha(p)
                for p in out.glob("private_*.joblib")
                if not p.name.startswith("._")
            },
        },
    )
    print("All704 predictions frozen before human criteria", flush=True)
    return predictions


def endpoints(g, columns):
    """Align original/reverse rows for complete identical author pairs."""
    a = g[g.order == "original"].set_index("author_id").sort_index()
    b = g[g.order == "reverse"].set_index("author_id").loc[a.index]
    return a.index, a[columns].to_numpy(), b[columns].to_numpy()


def summarize(out, predictions):
    """Inspect fidelity, native responses and criteria only after fitting freezes."""
    metrics, comparisons, links, teacher_links = [], [], [], []
    criterion_effects, target_contrasts = [], []
    labels = pd.read_csv(
        BASE / "all_factor_source_m29_v1/private_external_links.csv"
    ).set_index(["author_id", "query"])
    keys = ["query", "role", "input", "support"]
    for key, g in predictions.groupby(keys):
        metadata = dict(zip(keys, key, strict=True))
        for target in ("c", "f"):
            ids, a, b = endpoints(
                g, ["actual_" + target, "predicted_" + target, "native"]
            )
            for kind, v in (("original", a), ("even", (a + b) / 2), ("change", b - a)):
                metrics.append(
                    {
                        **metadata,
                        "target": target,
                        "kind": kind,
                        **paired_summary(v[:, 0], v[:, 1], calibrated=kind == "change"),
                        "rmse": float(np.sqrt(np.mean((v[:, 0] - v[:, 1]) ** 2))),
                        "native_r": correlation(v[:, 2], v[:, 1]),
                    }
                )
                if metadata["role"] == "source" and kind != "change":
                    truth = (
                        labels.xs(metadata["query"], level="query")
                        .loc[ids]
                        .y_true.to_numpy()
                    )
                    links.append(
                        {
                            **metadata,
                            "target": target,
                            "kind": kind,
                            **association(
                                v[:, 1], truth, binary=metadata["query"] == "TF_cont"
                            ),
                        }
                    )
    for (q, role, name), g in predictions.groupby(["query", "role", "input"]):
        for target in ("c", "f"):
            pairs = {}
            for support in ("natural", "paired"):
                ids, a, b = endpoints(
                    g[g.support == support], ["actual_" + target, "predicted_" + target]
                )
                pairs[support] = (a, b)
            for kind in ("original", "even", "change"):
                get = lambda a, b, kind=kind: (
                    a
                    if kind == "original"
                    else (a + b) / 2
                    if kind == "even"
                    else b - a
                )
                na, pa = (get(*pairs[k]) for k in ("natural", "paired"))
                comparisons.append(
                    {
                        "query": q,
                        "role": role,
                        "input": name,
                        "target": target,
                        "kind": kind,
                        **paired_delta(na[:, 0], pa[:, 1], na[:, 1], draws=1999),
                        "delta_mse": float(
                            np.mean(
                                (pa[:, 1] - pa[:, 0]) ** 2 - (na[:, 1] - na[:, 0]) ** 2
                            )
                        ),
                    }
                )
                if role == "source" and kind != "change":
                    truth = labels.xs(q, level="query").loc[ids].y_true.to_numpy()
                    criterion_effects.append(
                        {
                            "query": q,
                            "input": name,
                            "target": target,
                            "kind": kind,
                            **paired_delta(
                                truth,
                                pa[:, 1],
                                na[:, 1],
                                binary=q == "TF_cont",
                                draws=1999,
                            ),
                        }
                    )
    for q, g in predictions[
        (predictions.role == "source")
        & (predictions.input == "ordinal")
        & (predictions.support == "natural")
    ].groupby("query"):
        ids, a, b = endpoints(g, ["actual_c", "actual_f", "native"])
        y = labels.xs(q, level="query").loc[ids].y_true.to_numpy()
        for j, target in enumerate(("c", "f", "native")):
            target_contrasts.append(
                {
                    "query": q,
                    "target": target,
                    "comparison": "even_minus_original",
                    **paired_delta(
                        y,
                        (a[:, j] + b[:, j]) / 2,
                        a[:, j],
                        binary=q == "TF_cont",
                        draws=1999,
                    ),
                }
            )
            for kind, v in (
                ("original", a),
                ("even", (a + b) / 2),
                ("odd", (a - b) / 2),
            ):
                teacher_links.append(
                    {
                        "query": q,
                        "target": target,
                        "kind": kind,
                        **association(v[:, j], y, binary=q == "TF_cont"),
                    }
                )
    frames = {
        "fidelity": pd.DataFrame(metrics),
        "support_effects": pd.DataFrame(comparisons),
        "criterion_links": pd.DataFrame(links),
        "teacher_criterion_links": pd.DataFrame(teacher_links),
        "criterion_support_effects": pd.DataFrame(criterion_effects),
        "teacher_target_contrasts": pd.DataFrame(target_contrasts),
    }
    # The factorial interaction is defined on squared prediction error, not by
    # subtracting incomparable correlation coefficients from different targets.
    effects = frames["support_effects"]
    interactions = effects.pivot(
        index=["query", "role", "target", "kind"], columns="input", values="delta_mse"
    ).reset_index()
    interactions["interaction_delta_mse"] = interactions.early_pq - interactions.ordinal
    frames["interactions"] = interactions
    public = ROOT / "reports/grain_vector_mechanisms/paired_support_tables_v1"
    public.mkdir(exist_ok=True, parents=True)
    for name, frame in frames.items():
        frame.to_csv(out / (name + ".csv"), index=False)
        frame.to_csv(public / (name + ".csv"), index=False)
    design = read(out / "private_design.json")
    packetmeta = [
        read(out / "private_packets" / f"pair_{d['index']:03d}.json") for d in design
    ]
    provenance = {
        "report_script_sha256": sha(Path(__file__)),
        "status": read(out / "status.json"),
        "fit_freeze": read(out / "fit_freeze.json"),
        "max_old_replay_error": max(p["old_replay_error"] for p in packetmeta),
        "max_repeat_error": max(p.get("repeat_error", 0) for p in packetmeta),
        "token_bag_l1_max": max(d["token_bag_l1"] for d in design),
        "length_delta_range": [
            min(d["length_delta"] for d in design),
            max(d["length_delta"] for d in design),
        ],
        "table_hashes": {k: sha(public / (k + ".csv")) for k in frames},
    }
    write(public / "provenance.json", provenance)
    return frames


def main():
    """Execute the fixed factorial once; analysis outputs are reproducible."""
    with threadpool_limits(limits=2):
        if (OUT / "fit_freeze.json").exists():
            verify(OUT)
            freeze = read(OUT / "fit_freeze.json")
            if sha(OUT / "private_predictions.csv") != freeze["predictions_sha256"]:
                raise ValueError("Frozen predictions changed")
            for name, digest in freeze["model_hashes"].items():
                if sha(OUT / name) != digest:
                    raise ValueError("Frozen fitted model changed")
            predictions = pd.read_csv(OUT / "private_predictions.csv")
            print(
                "Reusing704 frozen predictions; no refit after criterion read",
                flush=True,
            )
        else:
            predictions = construct(OUT)
        frames = summarize(OUT, predictions)
    selected = frames["fidelity"].query("kind=='change' and role=='transport'")
    print(
        selected[
            [
                "query",
                "input",
                "support",
                "target",
                "r",
                "r_lower",
                "r_upper",
                "mse_over_zero",
                "native_r",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare fixed functional training losses using existing GRAIN/affinity panels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from grain_research.functional_reader import (
    ARMS,
    FunctionalCoordinates,
    author_panel_permutation,
    block_loss,
    build_blocks,
    fit_functional,
    operator_profile,
    stacked_system,
)
from grain_research.response_accounting import comment_average, paired_energy_interval
from grain_research.semantic_link import author_weights, weighted_metrics
from scripts.run_grain_response_accounting import correlation_columns, write

BASE = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1"
PRIOR, ACCOUNT = BASE / "semantic_link_r2", BASE / "response_accounting_r1"
PLAN = ROOT / "docs/GRAIN_FUNCTIONAL_READER_PLAN_V1.md"
ALPHAS = (0.0001, 0.001, 0.01, 0.1)
MODES = ("joint_linear", "joint_quadratic")
PARTS = ("level", "total", "within", "composition", "turnover")


def load_inputs():
    """Verify frozen input inventories; never deserialize questionnaire arrays."""
    for directory, key in ((PRIOR, "artifacts"), (ACCOUNT, "files")):
        summary = json.loads((directory / "summary.json").read_text())
        for name, sha in summary[key].items():
            if digest(directory / name) != sha:
                raise ValueError("Changed input: " + str(directory / name))
    with np.load(PRIOR / "private_inputs.npz", allow_pickle=False) as z:
        inputs = {
            k: z[k]
            for k in (
                "g",
                "z",
                "subreddits",
                "authors",
                "folds",
                "target",
                "author_ids",
            )
        }
    with np.load(ACCOUNT / "private_panels.npz", allow_pickle=False) as z:
        panel = {k: z[k] for k in ("folds", "counts", "conditions", "assignments")}
    with np.load(ACCOUNT / "private_accounting.npz", allow_pickle=False) as z:
        panel.update({k: z[k] for k in ("operators", "common", "strict")})
    addresses = pd.read_csv(PRIOR / "private_addresses.csv")
    if not np.array_equal(addresses.author_index, inputs["authors"]):
        raise ValueError("Address author mismatch")
    if not np.array_equal(panel["folds"][inputs["authors"]], inputs["folds"]):
        raise ValueError("Author fold mismatch")
    inputs["comment_ix"] = addresses.author_index.to_numpy(
        int
    ) * 36 + addresses.comment_index.to_numpy(int)
    actual = np.bincount(
        inputs["comment_ix"], minlength=len(panel["folds"]) * 36
    ).reshape(-1, 36)
    if not np.array_equal(actual, panel["counts"]):
        raise ValueError("Supported comments changed")
    inputs["names"] = json.loads((PRIOR / "contract.json").read_text())["target_names"]
    return inputs, panel


def as_panel(values, inputs, panel):
    """Average transformed sentence functions, not raw coordinates before squaring."""
    return comment_average(values, inputs["comment_ix"], panel["counts"].size)[
        0
    ].reshape(len(panel["folds"]), 36, -1)


def coordinate_panels(inputs, panel, source):
    """Fit new scalers on the specified source authors; transform without refitting."""
    smask = source[inputs["authors"]]
    coords = FunctionalCoordinates(inputs["mode"], 1.0).fit(
        inputs["g"][smask],
        inputs["z"][smask],
        inputs["subreddits"][smask],
        inputs["authors"][smask],
    )
    target_scaler = StandardScaler().fit(
        inputs["target"][smask], sample_weight=author_weights(inputs["authors"][smask])
    )
    x = as_panel(
        coords.design(inputs["g"], inputs["z"], inputs["subreddits"]), inputs, panel
    )
    y = as_panel(target_scaler.transform(inputs["target"]), inputs, panel)
    return coords, target_scaler, x, y


def blocks_for(x, y, panel, mask):
    """Restrict author panels before forming any learned sufficient statistics."""
    return build_blocks(
        x[mask],
        y[mask],
        panel["conditions"][mask],
        panel["assignments"][0, mask],
        panel["strict"][mask],
    )


def support_rows(blocks, mode, stage):
    """Expose the actual denominators and source normalizers of the objective."""
    return [
        {
            "map": mode,
            "stage": stage,
            "block": name,
            "authors": len(np.unique(b.authors)),
            "rows": len(b.y),
            "energy": b.energy,
        }
        for name, b in blocks.items()
    ]


def metric_rows(y, p, panel, mask, names, model, cohort):
    """Evaluate comment and functional predictions on identical supported authors."""
    rows = []
    ok = mask[:, None] & np.isfinite(y).all(2)
    authors = np.broadcast_to(np.arange(len(y))[:, None], ok.shape)[ok]
    r, mse, _ = weighted_metrics(y[ok], p[ok], authors, np.ones(y.shape[-1]))
    batches = [("comment", r, mse, len(np.unique(authors)))]
    yp = operator_profile(y, panel["operators"][0], panel["common"][0], panel["strict"])
    pp = operator_profile(p, panel["operators"][0], panel["common"][0], panel["strict"])
    for j, part in enumerate(PARTS):
        select = mask & np.isfinite(yp[:, j]).all(1)
        yy, pred = yp[select, j], pp[select, j]
        batches.append(
            (part, correlation_columns(yy, pred), ((yy - pred) ** 2).mean(0), len(yy))
        )
    for part, r, mse, n in batches:
        for q, name in enumerate(names):
            kind, facet = name.split(":")
            rows.append(
                {
                    "model": model,
                    "cohort": cohort,
                    "part": part,
                    "function": kind,
                    "facet": facet,
                    "authors": n,
                    "reader_teacher_r": r[q],
                    "mse": mse[q],
                }
            )
    return rows


def paired_rows(y, p, baseline, panel, held, names, model):
    """Author-cluster error differences; never compare differently supported rows."""
    yp, pp, bp = [
        operator_profile(a, panel["operators"][0], panel["common"][0], panel["strict"])
        for a in (y, p, baseline)
    ]
    outputs = []
    for kind in ("mean", "max"):
        qi = np.asarray([n.startswith(kind + ":") for n in names])
        for j, part in enumerate(PARTS):
            mask = held & np.isfinite(yp[:, j]).all(1)
            base_error = ((yp[mask, j][:, qi] - bp[mask, j][:, qi]) ** 2).mean(1)
            new_error = ((yp[mask, j][:, qi] - pp[mask, j][:, qi]) ** 2).mean(1)
            ci = paired_energy_interval(base_error, new_error, draws=500, seed=915)
            outputs.append(
                {
                    "model": model,
                    "function": kind,
                    "part": part,
                    "mse_improvement": ci.pop("delta_mean_energy"),
                    **ci,
                }
            )
    return outputs


def random_rows(y, p, panel, held, names, model):
    """Average repeated random-partition errors within each author first."""
    errors = []
    for r in range(1, len(panel["operators"])):
        yp, pp = [
            operator_profile(
                a, panel["operators"][r], panel["common"][r], np.ones(len(y), bool)
            )
            for a in (y, p)
        ]
        errors.append((yp - pp) ** 2)
    errors = np.stack(errors)
    rows = []
    for j, part in ((1, "total"), (2, "within")):
        mask = held & np.isfinite(errors[:, :, j]).all(axis=(0, 2))
        for kind in ("mean", "max"):
            qi = [k for k, name in enumerate(names) if name.startswith(kind + ":")]
            rows.append(
                {
                    "model": model,
                    "part": part,
                    "function": kind,
                    "authors": int(mask.sum()),
                    "random_mean_mse": float(errors[:, mask, j][:, :, qi].mean()),
                }
            )
    return rows


def run(args):
    """Fit fixed loss arms, freeze them, evaluate once, and save complete evidence."""
    out, start = args.output, time.monotonic()
    if out.exists():
        raise ValueError("Use a fresh output directory")
    out.mkdir(parents=True)
    try:
        with threadpool_limits(limits=2):
            inputs, panel = load_inputs()
            fold = panel["folds"]
            train, val, source, held = fold >= 2, fold == 1, fold != 0, fold == 0
            write(
                out / "contract.json",
                {
                    "plan_sha256": digest(PLAN),
                    "runner_sha256": digest(Path(__file__)),
                    "module_sha256": digest(
                        ROOT / "grain_research/functional_reader.py"
                    ),
                    "semantic_module_sha256": digest(
                        ROOT / "grain_research/semantic_link.py"
                    ),
                    "accounting_module_sha256": digest(
                        ROOT / "grain_research/response_accounting.py"
                    ),
                    "C14_summary_sha256": digest(PRIOR / "summary.json"),
                    "accounting_summary_sha256": digest(ACCOUNT / "summary.json"),
                    "alphas": ALPHAS,
                    "maps": MODES,
                    "arms": ARMS,
                    "selection": "fold1 point loss only, same selected alpha for all arms of map",
                    "source_fit_folds": [2, 3, 4],
                    "validation_fold": 1,
                    "refit_folds": [1, 2, 3, 4],
                    "held_fold": 0,
                    "historically_explored_development_holdout": True,
                    "target_names": inputs["names"],
                    "new_neural_calls": 0,
                    "questionnaire_arrays_accessed": False,
                    "seed": 915,
                    "bootstrap_draws": 500,
                    "weighting": "equal authors; equal candidate-supported comments per author",
                    "time_limit_seconds": 1200,
                },
            )
            tuning, selection, supports, diagnostics, metrics = [], {}, [], [], []
            predictions, snapshots = {}, {}
            final_y = None
            for mode in MODES:
                inputs["mode"] = mode
                coords, scaler, x, y = coordinate_panels(inputs, panel, train)
                bt, bv = blocks_for(x, y, panel, train), blocks_for(x, y, panel, val)
                supports += support_rows(bt, mode, "train")
                losses = []
                for alpha in ALPHAS:
                    beta, diag = fit_functional(bt, "point", alpha)
                    loss = block_loss(bv["point"], beta)
                    tuning.append(
                        {"map": mode, "alpha": alpha, "validation_point_mse": loss}
                    )
                    losses.append(loss)
                alpha = ALPHAS[int(np.argmin(losses))]
                selection[mode] = alpha
                for arm in ARMS:
                    beta, _ = fit_functional(bt, arm, alpha)
                    metrics += metric_rows(
                        y,
                        x @ beta,
                        panel,
                        val,
                        inputs["names"],
                        mode + ":" + arm,
                        "validation",
                    )
                coords, scaler, x, y = coordinate_panels(inputs, panel, source)
                if final_y is not None and not np.array_equal(
                    final_y, y, equal_nan=True
                ):
                    raise ValueError("Coordinate arms changed target units")
                final_y = y
                blocks = blocks_for(x, y, panel, source)
                supports += support_rows(blocks, mode, "source_refit")
                source_energies = {k: b.energy for k, b in blocks.items()}
                for arm in (*ARMS, "permuted_balanced"):
                    if time.monotonic() - start > 1200:
                        raise TimeoutError("Declared wall-time budget reached")
                    name, fit_arm = mode + ":" + arm, arm
                    fit_blocks = blocks
                    if arm == "permuted_balanced":
                        shuffled, order = author_panel_permutation(y[source])
                        fit_blocks = build_blocks(
                            x[source],
                            shuffled,
                            panel["conditions"][source],
                            panel["assignments"][0, source],
                            panel["strict"][source],
                        )
                        supports += support_rows(fit_blocks, mode, "permuted_source")
                        fit_arm = "balanced"
                        np.save(out / f"private_{mode}_permutation.npy", order)
                    beta, diag = fit_functional(fit_blocks, fit_arm, alpha)
                    artifact = {
                        "coordinates": coords,
                        "target_scaler": scaler,
                        "beta": beta,
                        "arm": arm,
                        "alpha": alpha,
                        "fit_energies": {k: b.energy for k, b in fit_blocks.items()},
                        "source_authors": np.flatnonzero(source),
                        "source_energies": source_energies,
                    }
                    path = out / (name.replace(":", "_") + ".joblib")
                    joblib.dump(artifact, path)
                    pred = x @ beta
                    if not np.array_equal(
                        pred, x @ joblib.load(path)["beta"], equal_nan=True
                    ):
                        raise ValueError("Frozen reader reload differs")
                    predictions[name] = pred
                    sx, sy = stacked_system(fit_blocks, fit_arm)
                    snapshots[name.replace(":", "_") + "_gram"] = sx.T @ sx
                    snapshots[name.replace(":", "_") + "_rhs"] = sx.T @ sy
                    diagnostics.append({"model": name, **diag})
                print(
                    json.dumps(
                        {"map": mode, "alpha": alpha, "frozen_models": len(predictions)}
                    ),
                    flush=True,
                )
            # Commit the source decision before accessing held metrics.
            write(
                out / "source_selection.json",
                {
                    "selected_alpha": selection,
                    "tuning": tuning,
                    "models": {p.name: digest(p) for p in out.glob("*.joblib")},
                },
            )
            pd.DataFrame(tuning).to_csv(out / "source_tuning.csv", index=False)
            pd.DataFrame(supports).to_csv(out / "source_support.csv", index=False)
            np.savez_compressed(out / "private_normal_equations.npz", **snapshots)
            np.savez_compressed(
                out / "private_predictions.npz",
                teacher=final_y,
                **{k.replace(":", "_"): v for k, v in predictions.items()},
            )
            paired, random = [], []
            for name, p in predictions.items():
                metrics += metric_rows(
                    final_y, p, panel, held, inputs["names"], name, "held"
                )
                base = predictions[name.split(":")[0] + ":point"]
                paired += paired_rows(
                    final_y, p, base, panel, held, inputs["names"], name
                )
                random += random_rows(final_y, p, panel, held, inputs["names"], name)
            frame = pd.DataFrame(metrics)
            frame.to_csv(out / "metrics_by_facet.csv", index=False)
            agg = frame.groupby(
                ["model", "cohort", "part", "function"], as_index=False
            ).agg(
                authors=("authors", "first"),
                mean_r=("reader_teacher_r", "mean"),
                mean_mse=("mse", "mean"),
            )
            agg.to_csv(out / "metrics_summary.csv", index=False)
            pd.DataFrame(paired).to_csv(out / "paired_improvement.csv", index=False)
            pd.DataFrame(random).to_csv(
                out / "random_partition_metrics.csv", index=False
            )
            write(out / "solver_audit.json", {"diagnostics": diagnostics})
            summary = {
                "status": "FUNCTIONAL_LOSS_COMPARISON_COMPLETE",
                "selected_alpha": selection,
                "source_authors": int(source.sum()),
                "held_authors": int(held.sum()),
                "frozen_models": len(predictions),
                "wall_seconds": time.monotonic() - start,
                "goal_complete": False,
                "scope": "fixed semantic-affinity functionals, not questionnaire or causal response",
                "files": {
                    p.name: digest(p)
                    for p in out.iterdir()
                    if p.is_file() and not p.name.startswith("._")
                },
            }
            write(out / "summary.json", summary)
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k != "files"}, indent=2
                ),
                flush=True,
            )
    except Exception as exc:
        (out / "error_log.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "functional_reader_r1")
    run(parser.parse_args())

#!/usr/bin/env python3
"""NEXT17/18: collect bounded conditional comment and retained-state quartets."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.comment_carrier import (
    joint_account,
    partition,
    reference_state,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.level_response import correlation
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_centered_adjoint_field import CELLS, SOURCE, verify
from scripts.run_grain_configuration_parity import anchor_coordinates
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

FIELD = BASE / "centered_adjoint_field_v1"
PLAN = ROOT / "docs/GRAIN_LOGIC_COMMENT_CARRIER_PLAN_V1.md"
PAIRS = {"adjacent": (0, 1), "distant": (0, 11)}


def prepare(out):
    """Bind cached state/layout hashes and the exposed cohort before collecting effects."""
    parent = verify(FIELD)
    design = read(FIELD / "private_design.json")
    if len(design) != 48 or any(len(r["views"][0]["groups"]) != 12 for r in design):
        raise ValueError("Exact old48-author/12-comment panel required")
    sources = {}
    for row in design:
        for p in (
            SOURCE / "private_packets" / f"author_{row['index']:03d}.npz",
            CELLS / "private_packets" / f"author_{row['index']:03d}.json",
        ):
            digest = sha(p)
            if digest != parent["sources"][str(p.relative_to(ROOT))]:
                raise ValueError("Old state or endpoint changed")
            sources[str(p.relative_to(ROOT))] = digest
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/comment_carrier.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
        ROOT / "grain_research/vector_mechanisms/context_carrier.py",
        ROOT / "scripts/run_grain_mean_null_path.py",
        ROOT / "scripts/run_grain_configuration_parity.py",
    ]
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", design)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COMMENT_CARRIER_LOGIC_V1",
            "design_sha256": sha(out / "private_design.json"),
            "field_manifest_sha256": sha(FIELD / "manifest.json"),
            "reference_sha256": parent["reference_sha256"],
            "model_sha256": parent["model_sha256"],
            "source_outcome_sha256": parent["source_outcome_sha256"],
            "sources": sources,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "max_calls": 420,
            "max_seconds": 600,
            "intended_calls": 410,
            "views": [0],
            "pairs": PAIRS,
            "human_labels_for_selection_or_fitting": False,
        },
    )
    write(out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0})
    print({"status": "PREPARED", "authors": 48, "intended_suffix_calls": 410})


def verify_run(out):
    """Verify inputs and code independently of the human-outcome file."""
    m = read(out / "manifest.json")
    for p, digest in [
        (out / "private_design.json", m["design_sha256"]),
        (FIELD / "manifest.json", m["field_manifest_sha256"]),
        (LEVEL / "reference.npz", m["reference_sha256"]),
        *[(ROOT / p, h) for p, h in m["code_hashes"].items()],
    ]:
        if sha(p) != digest:
            raise ValueError("Frozen comment/carrier protocol changed")
    return m


def collect(out):
    """Run suffix-only interventions; never change weights or load questionnaire Y."""
    m, prior = verify_run(out), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete; refusing duplicate collection")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Observer model changed")
    design = read(out / "private_design.json")
    donors = []
    for row in design:
        if row["query"] == "TF_cont" and row["role"] == "reference":
            p = SOURCE / "private_packets" / f"author_{row['index']:03d}.npz"
            if sha(p) != m["sources"][str(p.relative_to(ROOT))]:
                raise ValueError("Donor state changed")
            with np.load(p, allow_pickle=False) as a:
                v = row["views"][0]
                donors.append((a["view_0"].copy(), v["groups"], v["p_mask"]))
    if len(donors) != 4:
        raise ValueError("Four pre-existing references required")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        refs = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, old = observer.manifest(), read(FIELD / "observer.json")
    for k in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[k] != old[k]:
            raise ValueError("Observer execution changed")
    write(out / "observer.json", now)
    start, completed, seen = time.monotonic(), 0, set()
    device = next(observer.model.parameters()).device

    def status(value):
        """Save actual suffix cost and complete units, including failed runs."""
        payload = {
            "status": value,
            "authors": completed,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "prefix_calls": 0,
            "gradient_calls": 0,
            "training": False,
        }
        write(out / "run_status.json", payload)
        print(payload, flush=True)

    try:
        for row in design:
            query = row["query"]
            dst = out / "private_packets" / f"author_{row['index']:03d}.json"
            if dst.exists():
                packet = read(dst)
                if packet["author_id"] != row["author_id"] or packet["query"] != query:
                    raise ValueError("Resumed identity mismatch")
                completed += 1
                seen.add(query)
                continue
            needed = 7 + 3 * (query == "TF_cont") + int(query not in seen)
            if (
                prior["calls"] + observer.calls + needed > m["max_calls"]
                or prior["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            p = SOURCE / "private_packets" / dst.with_suffix(".npz").name
            cp = CELLS / "private_packets" / dst.name
            for source in (p, cp):
                if sha(source) != m["sources"][str(source.relative_to(ROOT))]:
                    raise ValueError("Frozen state/cell changed")
            v = row["views"][0]
            g, _, body, rest = partition(v["groups"], v["p_mask"])
            with np.load(p, allow_pickle=False) as a:
                h = torch.tensor(a["view_0"][None], device=device)
            native = anchor_coordinates(
                h, np.array(v["p_mask"], bool), refs[f"{query}__Z"], refs[f"{query}__h"]
            )
            base = native + mean_null_direction(native[0], g, list(range(12)))[None]
            unaffected = np.ones(native.shape[1], bool)
            unaffected[body] = False
            if not torch.equal(base[:, unaffected], native[:, unaffected]):
                raise ValueError("Collapse changed nonbody/PQ")
            mean_error = max(
                float((base[0, x].mean(0) - native[0, x].mean(0)).abs().max().cpu())
                for x in g
            )
            if mean_error > 2e-5:
                raise ValueError("Comment means not retained")
            kwargs = cached_suffix_kwargs(observer.model, native)
            scores = {
                "zero": replay_suffix(observer, base, kwargs, v["alphabet"]),
                "full": replay_suffix(observer, native, kwargs, v["alphabet"]),
            }
            cell = read(cp)["views"][0]
            replay_error = max(
                abs(scores["zero"]["conditional_score"] - cell["f00"]),
                abs(scores["full"]["conditional_score"] - cell["f10"]),
            )
            if replay_error > 2e-6:
                raise ValueError("Old fixed-query endpoint replay failed")
            repeat_error = None
            if query not in seen:
                again = replay_suffix(observer, base, kwargs, v["alphabet"])
                repeat_error = float(
                    np.max(
                        np.abs(
                            np.array(again["candidate_logits"])
                            - scores["zero"]["candidate_logits"]
                        )
                    )
                )
                if repeat_error > 2e-6:
                    raise ValueError("Observer not repeatable")
                seen.add(query)
            for name, selected in {
                "s0": [0],
                "s1": [1],
                "s11": [11],
                "adjacent": [0, 1],
                "distant": [0, 11],
            }.items():
                edited = base.clone()
                positions = np.concatenate([g[j] for j in selected])
                edited[:, positions] = native[:, positions]
                scores[name] = replay_suffix(observer, edited, kwargs, v["alphabet"])
            audit = {
                "replay_error": replay_error,
                "repeat_logit_error": repeat_error,
                "mean_error": mean_error,
                "tokens": len(h[0]),
                "body_tokens": len(body),
                "retained_tokens": len(rest),
            }
            if query == "TF_cont":
                ref, count = reference_state(
                    base[0].cpu().numpy(), g, v["p_mask"], donors
                )
                ref = torch.as_tensor(ref[None], device=device)
                keep = np.ones(native.shape[1], bool)
                keep[np.concatenate([body, rest])] = False
                if not torch.equal(ref[:, keep], base[:, keep]):
                    raise ValueError("Carrier reference changed P/Q")
                for name, b, r in (
                    ("b00", False, False),
                    ("b10", True, False),
                    ("b01", False, True),
                ):
                    edited = ref.clone()
                    if b:
                        edited[:, body] = base[:, body]
                    if r:
                        edited[:, rest] = base[:, rest]
                    scores[name] = replay_suffix(
                        observer, edited, kwargs, v["alphabet"]
                    )
                audit.update(
                    interpolated_roles=count,
                    body_edit_relative_norm=float(
                        (
                            (ref[:, body] - base[:, body]).norm() / base[:, body].norm()
                        ).cpu()
                    ),
                    retained_edit_relative_norm=float(
                        (
                            (ref[:, rest] - base[:, rest]).norm() / base[:, rest].norm()
                        ).cpu()
                    ),
                )
            write(
                dst,
                {
                    "author_id": row["author_id"],
                    "query": query,
                    "role": row["role"],
                    "alphabet": v["alphabet"],
                    "scores": scores,
                    "audit": audit,
                },
            )
            completed += 1
            if completed % 8 == 0:
                status("RUNNING")
        status("COMPLETE")
        write(
            out / "endpoint_freeze.json",
            {
                "human_outcomes_read": False,
                "packets": {
                    p.name: sha(p)
                    for p in sorted((out / "private_packets").glob("author_*.json"))
                },
            },
        )
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def construct(out):
    """Freeze scalar mechanism accounts before connecting any human criterion."""
    verify_run(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete bounded endpoints required")
    freeze = read(out / "endpoint_freeze.json")
    joint, carriers, quality = [], [], []
    for name, digest in freeze["packets"].items():
        p = out / "private_packets" / name
        if sha(p) != digest:
            raise ValueError("Endpoint changed")
        packet = read(p)
        common = {k: packet[k] for k in ("author_id", "query", "role")}
        scores = packet["scores"]
        for pair, (i, j) in PAIRS.items():
            part = joint_account(
                *(
                    scores[x]["candidate_logits"]
                    for x in ("zero", f"s{i}", f"s{j}", pair)
                ),
                packet["alphabet"]["values"],
            )
            if (
                abs(
                    part["interaction"]
                    - part["internal_score"]
                    - part["link_curvature"]
                )
                > 1e-12
            ):
                raise ValueError("Interaction account failed")
            joint.append({**common, "pair": pair, **part})
        if packet["query"] == "TF_cont":
            b11, b10, b01, b00 = [
                scores[x]["conditional_score"] for x in ("zero", "b10", "b01", "b00")
            ]
            interaction = b11 - b10 - b01 + b00
            carriers.append(
                {
                    **common,
                    "native": b11,
                    "baseline": b00,
                    "body": b10 - b00,
                    "retained": b01 - b00,
                    "interaction": interaction,
                    "body_allocated": b10 - b00 + interaction / 2,
                    "retained_allocated": b01 - b00 + interaction / 2,
                }
            )
        for name, r in scores.items():
            quality.append(
                {
                    **common,
                    "cell": name,
                    "score": r["conditional_score"],
                    "candidate_mass": r["candidate_mass"],
                    **packet["audit"],
                }
            )
    outputs = {
        "private_joint.csv": joint,
        "private_carriers.csv": carriers,
        "private_quality.csv": quality,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(out / name, index=False)
    write(
        out / "contrast_freeze.json",
        {
            "human_outcomes_read": False,
            "files": {name: sha(out / name) for name in outputs},
        },
    )
    print(
        {
            "status": "MODEL_ACCOUNTS_FROZEN",
            "joint_rows": len(joint),
            "carrier_rows": len(carriers),
        }
    )


def stats(v, y, total):
    """Describe one measured action with a common-root signed criterion contribution."""
    v, y, total = (np.asarray(a, float) for a in (v, y, total))
    denominator = total.std() * y.std()
    return {
        "mean": float(v.mean()),
        "sd": float(v.std()),
        "rms": float(np.sqrt(np.mean(v**2))),
        "r_y": correlation(v, y),
        "r_contribution": float(np.mean((v - v.mean()) * (y - y.mean())) / denominator)
        if denominator
        else float("nan"),
    }


def describe(data, column, root, seed):
    """Use fixed whole-author bootstrap draws; no fitting or outcome-selected arms."""
    x, y, total = [data[c].to_numpy() for c in (column, "y_true", root)]
    actual = stats(x, y, total)
    indices = np.random.default_rng(seed).integers(0, len(x), (1999, len(x)))
    draws = pd.DataFrame([stats(x[ix], y[ix], total[ix]) for ix in indices])
    result = dict(actual)
    for k in actual:
        finite = draws[k][np.isfinite(draws[k])]
        result.update(
            {
                f"{k}_low": float(finite.quantile(0.025)),
                f"{k}_high": float(finite.quantile(0.975)),
            }
        )
    return result


def analyze(out, name):
    """Join only the old permitted source outcomes after all numerical functions froze."""
    m = verify_run(out)
    frozen = read(out / "contrast_freeze.json")
    if frozen["human_outcomes_read"] is not False:
        raise ValueError("Expected outcome-free contrast freeze")
    for path, digest in frozen["files"].items():
        if sha(out / path) != digest:
            raise ValueError("Frozen numerical contrast changed")
    path = SOURCE / "private_anchor_source_connection.csv"
    if sha(path) != m["source_outcome_sha256"]:
        raise ValueError("Permitted source criterion changed")
    keys = ["author_id", "query"]
    y = pd.read_csv(path, usecols=keys + ["y_true"], dtype={"author_id": str})
    dest = out / name
    dest.mkdir(exist_ok=False)
    frames = {}
    for family in ("joint", "carriers"):
        data = pd.read_csv(out / f"private_{family}.csv", dtype={"author_id": str})
        data = data.merge(y, on=keys, how="left", validate="many_to_one")
        if not np.isfinite(data.y_true).all():
            raise ValueError("Missing criterion link")
        data.to_csv(dest / f"private_{family}_criterion.csv", index=False)
        frames[family] = data
    rows = []
    for qi, ((query, pair), g) in enumerate(frames["joint"].groupby(["query", "pair"])):
        g = g.sort_values("author_id")
        if len(g) != 24 or g.author_id.duplicated().any():
            raise ValueError("24 distinct authors per pair required")
        for component in (
            "pair_action",
            "effect_i",
            "effect_j",
            "interaction",
            "internal_score",
            "link_curvature",
            "logit_interaction_rms",
            "logit_pair_action_rms",
        ):
            rows.append(
                {
                    "query": query,
                    "pair": pair,
                    "component": component,
                    "n": len(g),
                    **describe(g, component, "pair_action", 20260910 + qi),
                }
            )
    pd.DataFrame(rows).to_csv(dest / "joint_metrics.csv", index=False)
    rows = []
    for scope in ("reference_external", "all"):
        g = frames["carriers"].sort_values("author_id")
        if scope == "reference_external":
            g = g[g.role == "evaluation"]
        if len(g) != (20 if scope == "reference_external" else 24):
            raise ValueError("Carrier reference boundary changed")
        for component in (
            "native",
            "baseline",
            "body",
            "retained",
            "interaction",
            "body_allocated",
            "retained_allocated",
        ):
            rows.append(
                {
                    "query": "TF_cont",
                    "scope": scope,
                    "component": component,
                    "n": len(g),
                    **describe(g, component, "native", 20260910),
                }
            )
    pd.DataFrame(rows).to_csv(dest / "carrier_metrics.csv", index=False)
    q = pd.read_csv(out / "private_quality.csv")
    q.groupby(["query", "cell"], sort=True).agg(
        n=("score", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        score_sd=("score", "std"),
        mass_min=("candidate_mass", "min"),
        mass_mean=("candidate_mass", "mean"),
    ).reset_index().to_csv(dest / "quality.csv", index=False)
    closure = np.max(
        np.abs(
            frames["carriers"].native
            - frames["carriers"][["baseline", "body", "retained", "interaction"]].sum(
                axis=1
            )
        )
    )
    summary = {
        "status": "COMPLETE",
        "cost": read(out / "run_status.json"),
        "old_endpoint_replay_max": float(q.replay_error.max()),
        "repeat_logit_error_max": float(q.repeat_logit_error.max()),
        "mean_error_max": float(q.mean_error.max()),
        "carrier_closure_error": float(closure),
        "interpolated_roles": int(q.interpolated_roles.fillna(0).max()),
        "body_edit_relative_norm_mean": float(q.body_edit_relative_norm.mean()),
        "retained_edit_relative_norm_mean": float(q.retained_edit_relative_norm.mean()),
        "author_counts": {"TF_cont": 24, "neuroticism": 24},
        "criterion_fitting": False,
        "question": "conditional mechanism, not personality prediction accuracy",
        "numerical_hashes": {p.name: sha(p) for p in sorted(dest.glob("*.csv"))},
    }
    write(dest / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "numerical_hashes"})


def main():
    """Run a single finite protocol without resurrecting any historical queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "construct", "analyze"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_comment_carrier_v1"
    )
    parser.add_argument("--analysis-name", default="analysis_r1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.analysis_name.replace("_", "").isalnum()
    ):
        raise ValueError("Use an ignored output directory and simple analysis name")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    if args.phase == "analyze":
        analyze(out, args.analysis_name)
    else:
        {"prepare": prepare, "collect": collect, "construct": construct}[args.phase](
            out
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Locate task-relative observation loss on the SAME natural event interface."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_trait_action as E
from grain_research.construction import math_profile_reader as P
from grain_research.construction import math_target_complement as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_target_complement_hm184.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_observation_complement_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_COMPLEMENT_PLAN_HM184.md",
    "grain_research/construction/math_target_complement.py",
    "scripts/run_grain_target_complement_hm184.py",
    "scripts/score_grain_target_complement_cold.py",
    "tests/test_grain_target_complement.py",
    "grain_research/construction/math_event_trait_action.py",
    "grain_research/construction/math_profile_reader.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def arrays(path):
    """Read registered numerical arrays, never raw text or questionnaires."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cosine(x, y):
    """Preserve undefined zero directions instead of scoring them as success."""
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.ravel(x) @ np.ravel(y) / den) if den else None


def prepare(out):
    """Freeze actual evidence identities and method before new response analysis."""
    cfg = R.read(ROOT / CONFIG)
    vp, ep, jp = (
        ROOT / cfg[k] for k in ("vector_parent", "event_parent", "joint_parent")
    )
    parents = (
        (vp, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")),
        (ep, ("compiled_seal.json", "analysis_seal.json")),
        (jp, ("scored_seal.json", "analysis_seal.json")),
    )
    inputs, code = {}, {p: R.digest(ROOT / p) for p in SCIENCE}
    for parent, stages in parents:
        manifest = R.check(parent, stages)
        code.update(manifest["code"])
        for name in ("manifest.json", "prepared_seal.json", *stages):
            names = (
                (name,) if name == "manifest.json" else (name, *R.read(parent / name))
            )
            for item in names:
                path = str((parent / item).relative_to(ROOT))
                inputs[path] = R.digest(ROOT / path)
    v, e = R.prepared(vp), R.prepared(ep)
    joint = R.prepared(jp)
    if any(
        not np.array_equal(joint[k], e[k])
        for k in ("author_ids", "selected", "control")
    ):
        raise ValueError("Native arrays belong to different event identities")
    if not np.array_equal(v["source"], e["source_indices"]) or not np.array_equal(
        v["panel"], e["application_indices"]
    ):
        raise ValueError("Different source or application author families")
    if set(v["source"]) & set(v["panel"]):
        raise ValueError("Source and application author overlap")
    with np.load(ROOT / cfg["material"], allow_pickle=False) as z:
        source = E.unit_vectors(z["comments"][v["source"], :24]).reshape(
            386, 2, 12, 1024
        )
    with np.load(ROOT / cfg["directions"], allow_pickle=False) as z:
        mapping = np.column_stack(
            [
                z[k]
                for k in (
                    "author_stability",
                    "response_stability",
                    "principal_components",
                )
            ]
        )
    if (
        np.max(np.abs(source @ mapping - v["source_coordinates"]))
        > cfg["comparison_tolerance"]
    ):
        raise ValueError("Changed source1024->15 vector identities")
    old_manifest = R.read(ep / "manifest.json")
    old_cfg = old_manifest["config"]
    d2 = ROOT / old_cfg["intervention_parent"]
    dm = R.read(d2 / "manifest.json")
    with np.load(d2 / "private_profiles.npz", allow_pickle=False) as z:
        poles = E.unit_vectors(z["query_poles"][2:4])
    query = poles[1] - poles[0]
    panel = []
    cache_paths = {Path(p).stem: p for p in dm["encoder_cache_sha256"]}
    for uid in e["author_ids"]:
        path = ROOT / cache_paths[str(uid)]
        if R.digest(path) != dm["encoder_cache_sha256"][str(path.relative_to(ROOT))]:
            raise ValueError("Changed historical comment vectors")
        aliases = R.read(path.with_suffix(".json"))["feature_aliases"]
        with np.load(path, allow_pickle=False) as z:
            panel.append(
                np.stack(
                    [
                        E.unit_vectors(z[aliases.get(k, k)][:2])
                        for k in ("cue_mask_comment_vectors", "raw_comment_vectors")
                    ]
                )
            )
        for p in (path, path.with_suffix(".json")):
            inputs[str(p.relative_to(ROOT))] = R.digest(p)
    panel = np.asarray(panel)
    replay = float(np.max(np.abs(panel[:, 0] @ query - e["signed_poles"])))
    coordinate_replay = float(np.max(np.abs(panel @ mapping - e["coordinates"])))
    if max(replay, coordinate_replay) > cfg["comparison_tolerance"]:
        raise ValueError("Changed SAME native event or semantic query")
    for path in (
        cfg["material"],
        cfg["directions"],
        str((d2 / "private_profiles.npz").relative_to(ROOT)),
    ):
        inputs[path] = R.digest(ROOT / path)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": code,
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source_vectors=source,
        mapping=mapping,
        query=query,
        labels=v["targets"],
        source_indices=v["source"],
        panel_indices=v["panel"],
        selected=e["selected"],
        control=e["control"],
        panel_vectors=panel,
    )
    count = 0
    with (out / "cold_input.jsonl").open("x") as sink:
        for u, arm, view in np.ndindex(24, 2, 2):
            ss, cc, all_slots = (
                e["selected"][u, view],
                e["control"][u, view],
                np.arange(12),
            )
            groups = (
                all_slots,
                all_slots[~np.isin(all_slots, ss)],
                all_slots[~np.isin(all_slots, cc)],
                ss,
                cc,
            )
            for k, slots in enumerate(groups):
                row = {
                    "frame_key": f"{arm}/{u}/{view}/{k}",
                    "events": [
                        {
                            "event_id": f"{u}:{view}:{int(t)}",
                            "slot": int(t),
                            "vector": panel[u, arm, view, t].tolist(),
                        }
                        for t in slots
                    ],
                }
                sink.write(json.dumps(row) + "\n")
                count += 1
    if count != cfg["cold_frames"]:
        raise ValueError("Incomplete cold scope")
    R.write(
        out / "quality.json",
        {
            "source_authors": 386,
            "panel_authors": 24,
            "cold_frames": count,
            "semantic_replay_error": replay,
            "coordinate_replay_error": coordinate_replay,
            "new_neural_API_private_application_labels": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "quality.json"),
    )
    print(json.dumps(R.read(out / "quality.json")))


def compile_source(out):
    """Construct all source fields and query companions without native responses."""
    R.check(out)
    z, start = R.prepared(out), time.perf_counter()
    q, triangular = M.fixed_basis(z["mapping"])
    fields = np.stack(
        [
            P.construct(z["source_vectors"][:, view].mean(1), z["labels"])["fields"]
            for view in (0, 1)
        ]
    )
    split = M.split_fields(fields, q)
    semantic = M.split_fields(z["query"], q)
    beta = M.coordinate_fields(fields, q, triangular)
    sbeta = M.coordinate_fields(z["query"], q, triangular)
    R.archive(
        out / "reader_bundle.npz",
        fields=split,
        semantic=semantic,
        version=np.array(M.VERSION),
    )
    R.archive(
        out / "coordinate_bundle.npz",
        source_beta=beta,
        semantic_beta=sbeta,
        mapping=z["mapping"],
        basis=q,
    )
    rows = []
    for view, target, component in np.ndindex(2, 5, 4):
        full, retained, complement = split[view, target, component]
        energy = float(full @ full)
        rows.append(
            {
                "view": view,
                "target": P.TARGETS[target],
                "component": P.COMPONENTS[component],
                "full_norm": float(np.linalg.norm(full)),
                "retained_squared_norm_ratio": float(retained @ retained / energy)
                if energy
                else None,
                "complement_squared_norm_ratio": float(complement @ complement / energy)
                if energy
                else None,
                "query_full_cosine": cosine(full, z["query"]),
                "query_retained_cosine": cosine(retained, semantic[1]),
                "query_complement_cosine": cosine(complement, semantic[2]),
                "full_cross_view_cosine": cosine(
                    fields[0, target, component], fields[1, target, component]
                ),
                "complement_cross_view_cosine": cosine(
                    split[0, target, component, 2], split[1, target, component, 2]
                ),
            }
        )
    R.table(out / "all_field_geometry.csv", rows)
    receipt = {
        "source_fields": 120,
        "query_fields": 3,
        "source_profile_decomposition_error": float(
            np.max(np.abs(fields[:, :, 0] - fields[:, :, 1:].sum(2)))
        ),
        "linear_span_reconstruction_error": float(
            np.max(np.abs(beta @ z["mapping"].T - split[..., 1, :]))
        ),
        "semantic_retained_squared_norm_ratio": float(
            semantic[1] @ semantic[1] / (semantic[0] @ semantic[0])
        ),
        "orthogonality_error": float(np.max(np.abs(split[..., 2, :] @ q))),
        "seconds": time.perf_counter() - start,
        "native_answers_read": False,
        "ratio_is_not_psychological_fraction": True,
    }
    if (
        max(
            receipt[k]
            for k in (
                "source_profile_decomposition_error",
                "linear_span_reconstruction_error",
                "orthogonality_error",
            )
        )
        > 2e-12
    ):
        raise ValueError("Registered source construction account failed")
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "reader_bundle.npz",
            "coordinate_bundle.npz",
            "all_field_geometry.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Launch a source-label-free/model-free cold arithmetic process once."""
    R.check(out, ("compiled_seal.json",))
    command = [
        sys.executable,
        str(ROOT / "scripts/score_grain_target_complement_cold.py"),
        "--bundle",
        str(out / "reader_bundle.npz"),
        "--input",
        str(out / "cold_input.jsonl"),
        "--output",
        str(out / "cold_output.jsonl"),
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120
    )
    receipt = json.loads(completed.stdout)
    if receipt["records"] != 480:
        raise ValueError("Incomplete cold observations")
    R.write(out / "cold_receipt.json", receipt)
    R.seal(out, "scored_seal.json", ("cold_output.jsonl", "cold_receipt.json"))
    print(completed.stdout.strip())


def all_readings(out):
    """Restore every addressed result without selecting a target or text view."""
    source, semantic = np.zeros((2, 24, 2, 5, 2, 5, 4, 3)), np.zeros((2, 24, 2, 5, 3))
    for line in (out / "cold_output.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = tuple(map(int, row["frame_key"].split("/")))
        source[key], semantic[key] = row["source"], row["semantic"]
    return source, semantic


def analyze(out):
    """Open historical native arrays only AFTER all frozen cold values exist."""
    cfg = R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"]
    src, sem = all_readings(out)
    old = arrays(ROOT / cfg["event_parent"] / "action_accounts.npz")
    joint = arrays(ROOT / cfg["joint_parent"] / "composition_accounts.npz")
    logits = joint["native_logits"]
    odds = logits[..., 1:] - logits[..., :1]
    natural_odds = odds[:, :, 1] - odds[:, :, 2]
    endpoints = np.concatenate(
        (
            old["native_effects"][:, :, :1],
            natural_odds,
            old["native_effects"][:, :, 1:3],
        ),
        axis=-1,
    )
    names = [
        "natural_ordinal",
        *[f"natural_digit{k}_vs0" for k in range(1, 10)],
        "halfpatch_ordinal",
        "wholepatch_ordinal",
    ]
    source_action = src[:, :, :, 1] - src[:, :, :, 2]
    semantic_action = sem[:, :, :, 1] - sem[:, :, :, 2]
    rows = []
    for arm, sv, target, comp, space, endpoint in np.ndindex(2, 2, 5, 4, 3, 12):
        for group in ("author_mean", "view0", "view1"):
            x, y = (
                source_action[arm, :, :, sv, target, comp, space],
                endpoints[:, :, endpoint],
            )
            if group == "author_mean":
                x, y = x.mean(1), y.mean(1)
            else:
                v = int(group[-1])
                x, y = x[:, v], y[:, v]
            rows.append(
                {
                    "reader": "source",
                    "representation": cfg["representations"][arm],
                    "source_view": sv,
                    "target": P.TARGETS[target],
                    "component": P.COMPONENTS[comp],
                    "space": M.SPACES[space],
                    "endpoint": names[endpoint],
                    "group": group,
                    "pearson": E.pearson(x, y),
                    "unscaled_cosine": cosine(x, y),
                    "reader_RMS": float(np.sqrt(np.mean(x**2))),
                    "native_RMS": float(np.sqrt(np.mean(y**2))),
                }
            )
    for arm, space, endpoint in np.ndindex(2, 3, 12):
        for group in ("author_mean", "view0", "view1"):
            x, y = semantic_action[arm, :, :, space], endpoints[:, :, endpoint]
            if group == "author_mean":
                x, y = x.mean(1), y.mean(1)
            else:
                v = int(group[-1])
                x, y = x[:, v], y[:, v]
            rows.append(
                {
                    "reader": "query_semantic",
                    "representation": cfg["representations"][arm],
                    "source_view": "",
                    "target": "N",
                    "component": "query_poles",
                    "space": M.SPACES[space],
                    "endpoint": names[endpoint],
                    "group": group,
                    "pearson": E.pearson(x, y),
                    "unscaled_cosine": cosine(x, y),
                    "reader_RMS": float(np.sqrt(np.mean(x**2))),
                    "native_RMS": float(np.sqrt(np.mean(y**2))),
                }
            )
    R.table(out / "all_native_contacts.csv", rows)
    account_error = float(
        np.max(np.abs(source_action[..., 0] - source_action[..., 1:].sum(-1)))
    )
    query_account_error = float(
        np.max(np.abs(semantic_action[..., 0] - semantic_action[..., 1:].sum(-1)))
    )
    semantic_replay = float(
        np.max(np.abs(semantic_action[0, ..., 0] - old["mathematical"][0, ..., 10, 0]))
    )
    summary = {
        "native_links": len(rows),
        "source_action_account_error": account_error,
        "query_action_account_error": query_account_error,
        "old_semantic_action_replay_error": semantic_replay,
        "semantic_main": [
            r
            for r in rows
            if r["reader"] == "query_semantic"
            and r["representation"] == "cue_mask"
            and r["endpoint"] == "natural_ordinal"
            and r["group"] == "author_mean"
        ],
        "source_N_main": [
            r
            for r in rows
            if r["reader"] == "source"
            and r["representation"] == "cue_mask"
            and r["target"] == "N"
            and r["component"] == "matched"
            and r["endpoint"] == "natural_ordinal"
            and r["group"] == "author_mean"
        ],
        "no_native_fit_or_prediction_promotion": True,
        "original_scope_complete": False,
    }
    if max(account_error, query_account_error, semantic_replay) > 2e-12:
        raise ValueError("SAME-event action account failed")
    R.archive(
        out / "action_accounts.npz",
        source_action=source_action,
        semantic_action=semantic_action,
        native_endpoints=endpoints,
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        ("all_native_contacts.csv", "action_accounts.npz", "summary.json"),
    )
    print(json.dumps(summary))


def verify(out):
    """Replay every cold score by literal per-event sums and original coordinates."""
    R.check(out, ("compiled_seal.json", "scored_seal.json", "analysis_seal.json"))
    bundle, coordinate = (
        arrays(out / "reader_bundle.npz"),
        arrays(out / "coordinate_bundle.npz"),
    )
    outputs = {
        r["frame_key"]: r
        for r in map(json.loads, (out / "cold_output.jsonl").read_text().splitlines())
    }
    direct_error = coordinate_error = 0.0
    scalars = 0
    for line in (out / "cold_input.jsonl").read_text().splitlines():
        row = json.loads(line)
        x = np.array([e["vector"] for e in row["events"]])
        fields, semantic = bundle["fields"], bundle["semantic"]
        direct = np.mean([fields @ e for e in x], axis=0)
        sem = np.mean([semantic @ e for e in x], axis=0)
        target = outputs[row["frame_key"]]
        direct_error = max(
            direct_error,
            float(np.max(np.abs(direct - target["source"]))),
            float(np.max(np.abs(sem - target["semantic"]))),
        )
        scores = x.mean(0) @ coordinate["mapping"]
        legacy = np.einsum("vqcd,d->vqc", coordinate["source_beta"], scores)
        coordinate_error = max(
            coordinate_error,
            float(np.max(np.abs(legacy - direct[..., 1]))),
            float(abs(scores @ coordinate["semantic_beta"] - sem[1])),
        )
        scalars += direct.size + sem.size
    receipt = {
        "cold_scalars": scalars,
        "literal_event_score_error": direct_error,
        "original_coordinate_score_error": coordinate_error,
        "same_agent_not_blind": True,
        "new_model_calls": 0,
    }
    if max(direct_error, coordinate_error) > 2e-12:
        raise ValueError("Cold alternate arithmetic failed")
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "compile", "score", "analyze", "verify")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {
        "prepare": prepare,
        "compile": compile_source,
        "score": score,
        "analyze": analyze,
        "verify": verify,
    }[args.stage](args.output)

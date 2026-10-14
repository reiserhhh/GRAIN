#!/usr/bin/env python3
"""Connect source-qualified author integration to identical evidence actions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_observation_channel as M
from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_source_meaning_bridge as B
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_author_observation_channel_hm234.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_observation_channel_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_OBSERVATION_CHANNEL_PLAN_HM234.md",
    "grain_research/construction/math_author_observation_channel.py",
    "grain_research/construction/math_joint_evidence.py",
    "grain_research/construction/math_source_meaning_bridge.py",
    "scripts/run_grain_author_observation_channel_hm234.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_author_observation_channel.py",
)


def check(out, stages=()):
    """Verify unchanged scientific inputs, implementation and completed stages."""
    return R.check(out, stages)


def prepare(out):
    """Bind source/window laws and current events before new outcome calculation."""
    cfg = R.read(ROOT / CONFIG)
    parents = {k: ROOT / cfg[k] for k in cfg if k.endswith("_parent")}
    bridge = parents["meaning_bridge_parent"]
    source = parents["source_bundle_parent"]
    windows = parents["source_windows_parent"]
    index = parents["source_index_parent"]
    S.check(bridge, ("cold_seal.json", "analyzed_seal.json", "verification_seal.json"))
    files = []
    specifications = (
        (
            bridge,
            "prepared_seal.json",
            ("observations.npz", "source_fields.json", "bindings.json"),
        ),
        (bridge, "cold_seal.json", ("cold_scores.npz",)),
        (source, "prepared_seal.json", ("source_bundle.json",)),
        (windows, "prepared_seal.json", ("observations.npz",)),
        (index, "prepared_seal.json", ("observations.npz",)),
        (parents["action_parent"], "analyzed_seal.json", ("analysis.npz",)),
        (parents["deletion_parent"], "analyzed_seal.json", ("analysis.npz",)),
    )
    for parent, seal, names in specifications:
        S.parent_seal(parent, seal, names)
        files.extend(str((parent / name).relative_to(ROOT)) for name in (seal,) + names)
    z = S.selected(
        bridge / "observations.npz",
        ("cells", "masks", "edges", "source_indices", "application_indices"),
    )
    original = S.selected(index / "observations.npz", ("source_indices", "source_mass"))
    observed = S.selected(windows / "observations.npz", ("mass", "counts"))
    bundle = R.read(source / "source_bundle.json")
    p, y = J.validate_bundle(bundle)
    ix = z["source_indices"]
    if (
        not np.array_equal(ix, original["source_indices"])
        or not np.array_equal(p, original["source_mass"])
        or len(ix) != cfg["source_authors"]
        or np.intersect1d(ix, z["application_indices"]).size
        or z["cells"].shape != (24, 12)
        or z["edges"].shape != (32, 2)
    ):
        raise ValueError("SOURCE386/current24 identity or law alignment changed")
    pw, counts = observed["mass"][ix], observed["counts"][ix]
    if pw.shape != (386, 2, 8) or not np.array_equal(counts, np.full((386, 2), 17)):
        raise ValueError("Original disjoint17-observation window catalogues required")
    if np.max(abs(pw.mean(1) - p)) > 2e-12:
        raise ValueError("Two-window garbling is not the inherited SOURCE channel")
    old = R.read(ROOT / cfg["historical_permutation_config"])
    perm = np.random.default_rng(old["permutation_seed"]).permutation(len(y))
    bundles = {"actual": bundle, "permuted": J.build_bundle(p, y[perm])}
    fields = R.read(bridge / "source_fields.json")
    for name, b in bundles.items():
        if (
            np.max(
                abs(np.asarray(b["event_probability"]) - fields[name]["probabilities"])
            )
            > 2e-12
            or np.max(abs(np.asarray(b["prior"]) - fields[name]["prior"])) > 2e-12
        ):
            raise ValueError("Old actual/permuted SOURCE meanings did not replay")
    bindings = R.read(bridge / "bindings.json")
    for u, binding in enumerate(bindings):
        if binding["author"] != u or binding["original_author_index"] != int(
            z["application_indices"][u]
        ):
            raise ValueError("Current event ownership changed")
        M.history_groups(z["cells"][u], z["masks"][u], binding["event_ids"])
    files.append(cfg["historical_permutation_config"])
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {path: R.digest(ROOT / path) for path in SCIENCE},
            "inputs": {path: R.digest(ROOT / path) for path in files},
            "historical_geometry_overlap": 17,
            "fresh_representation_holdout": False,
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **z, source_windows=pw, source_counts=counts)
    R.write(out / "source_bundles.json", bundles)
    R.write(out / "bindings.json", bindings)
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "unique_events": 288,
            "source_authors": 386,
            "edges": 32,
            "source_application_overlap": 0,
            "source_payload": "HM174 serialized SOURCE-only P/Y,not a questionnaire read",
            "historical_permutation_reused": old["permutation_seed"],
            "new_neural_API_private_training_application_label_calls": 0,
            "source_window_observations_per_author": counts.tolist(),
            "pair_design": "independent finite selectors,NOT independent natural occasions",
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "source_bundles.json",
            "bindings.json",
            "input_quality.json",
        ),
    )
    print("HM234 prepared;SAME288events/386SOURCE,old joint laws reused", flush=True)


def clean(x):
    """Serialize missing scientific values as null, never zero."""
    a = np.asarray(x)
    return np.where(np.isfinite(a), a, None).tolist()


def score(out):
    """Freeze all author scores before loading native/crossed model actions."""
    check(out)
    z = S.selected(out / "observations.npz", ("cells", "masks"))
    bundles, bindings = (
        R.read(out / "source_bundles.json"),
        R.read(out / "bindings.json"),
    )
    arrays, records = {}, []
    for name, bundle in bundles.items():
        probabilities = np.empty((24, 5, 3, 5))
        odds, supported = np.empty((24, 5, 2, 5)), np.empty((24, 5, 2, 5), bool)
        weights = np.empty((24, 5, 386))
        for u, binding in enumerate(bindings):
            groups = M.history_groups(
                z["cells"][u], z["masks"][u], binding["event_ids"]
            )
            for group, (cells, ids) in enumerate(groups):
                value = M.cold_history(bundle, cells, ids)
                probabilities[u, group] = value["probabilities"]
                odds[u, group] = value["binary_log_odds"]
                supported[u, group] = value["posterior_supported"]
                weights[u, group] = value["author_weights"]
                records.append(
                    {
                        "law": name,
                        "author": u,
                        "group": group,
                        "event_ids": ids,
                        "counts": J.counts(cells).tolist(),
                        "probabilities": clean(value["probabilities"]),
                        "posterior_supported": value["posterior_supported"].tolist(),
                        "effective_source_authors": value["effective_source_authors"],
                    }
                )
        arrays[name + "_probabilities"] = probabilities
        arrays[name + "_log_odds"] = odds
        arrays[name + "_supported"] = supported
        arrays[name + "_author_weights"] = weights
    R.archive(out / "cold_scores.npz", **arrays)
    R.write(out / "cold_records.json", records)
    R.seal(out, "cold_seal.json", ("cold_scores.npz", "cold_records.json"))
    print("HM234 cold scores saved:all240histories/3rules/5targets", flush=True)


def contacts(x, t):
    """Keep all meaning/action contacts and explicit missing-history counts."""
    x, t = np.asarray(x), np.asarray(t)
    shape = (5,) + t.shape[1:]
    result, valid = np.full(shape, np.nan), np.zeros(5, int)
    for q in range(5):
        keep = np.isfinite(x[:, q])
        valid[q] = int(keep.sum())
        if keep.any():
            result[q] = np.tensordot(x[keep, q], t[keep], axes=(0, 0)) / keep.sum()
    return result, valid


def analyze(out):
    """Compare fixed observation laws and complete signed meaning/action bridges."""
    started = time.monotonic()
    cfg = check(out, ("cold_seal.json",))["config"]
    z = S.selected(out / "observations.npz", ("edges", "source_windows"))
    bundles = R.read(out / "source_bundles.json")
    with np.load(out / "cold_scores.npz", allow_pickle=False) as f:
        cold = {k: f[k] for k in f.files}
    p = np.asarray(bundles["actual"]["source_mass"])
    q, garbling = M.window_pair(z["source_windows"][:, 0], z["source_windows"][:, 1])
    channels = {"single_event": M.channel(p), "two_window_pair": M.channel(q)}
    gain = (
        channels["two_window_pair"]["round_trip"]
        - channels["single_event"]["round_trip"]
    )
    gain_eig = np.linalg.eigvalsh(gain)
    if gain_eig[0] < -2e-12 or any(
        c["centered_spectrum"].max() > 1 + 2e-12 for c in channels.values()
    ):
        raise ValueError("Finite channel order failed;no inverse/precision rescue")
    arrays = {
        "pair_mass": q,
        "pair_garbling": garbling,
        "refinement_operator": gain,
        "refinement_eigenvalues": gain_eig,
    }
    for name, c in channels.items():
        for key, value in c.items():
            arrays[name + "_" + key] = value
    source_rows, target_grams = [], {}
    for law, bundle in bundles.items():
        target_grams[law] = {}
        for name, data in (("single_event", p), ("two_window_pair", q)):
            a = M.target_accounts(data, bundle["source_binary_targets"])
            for key, value in a.items():
                arrays[f"{law}_{name}_{key}"] = value
            target_grams[law][name] = {
                key: a[key].tolist()
                for key in ("label_gram", "field_gram", "return_gram")
            }
            for t, target in enumerate(cfg["targets"]):
                c0, c1, c2 = (
                    float(a[key][t, t])
                    for key in ("label_gram", "field_gram", "return_gram")
                )
                source_rows.append(
                    {
                        "law": law,
                        "channel": name,
                        "target": target,
                        "source_label_energy": c0,
                        "field_energy": c1,
                        "return_energy": c2,
                        "field_over_label": c1 / c0 if c0 else None,
                        "return_over_field": c2 / c1 if c1 else None,
                    }
                )
    action = S.selected(
        ROOT / cfg["action_parent"] / "analysis.npz",
        ("matrix", "interaction", "native", "native_increment"),
    )
    deletion = S.selected(ROOT / cfg["deletion_parent"] / "analysis.npz", ("raw",))
    modes = {
        "total": action["matrix"],
        "source_discrimination": action["matrix"]
        - action["matrix"].mean(2, keepdims=True),
        "reader_source_interaction": action["interaction"],
    }
    rows, direct_rows, connection_summary = [], [], {}
    for law, bundle in bundles.items():
        probs = cold[law + "_probabilities"]
        connection_summary[law] = {}
        for r, rule in enumerate(cfg["rules"]):
            value = probs[:, :, r]
            x = value[:, 0][z["edges"][:, 0]] - value[:, 0][z["edges"][:, 1]]
            arrays[f"{law}_{rule}_edge_scores"] = x
            for mode, t in modes.items():
                k, valid = contacts(x, t)
                arrays[f"{law}_{rule}_{mode}"] = k
                for target, reader, encoding in np.ndindex(5, 5, 5):
                    rows.append(
                        {
                            "law": law,
                            "rule": rule,
                            "action": mode,
                            "meaning": cfg["targets"][target],
                            "reader": cfg["targets"][reader],
                            "encoding": cfg["targets"][encoding],
                            "contact": clean(k[target, reader, encoding]),
                            "valid_edges": int(valid[target]),
                            "missing_edges": 32 - int(valid[target]),
                        }
                    )
            native, _ = contacts(x, action["native"])
            increment, _ = contacts(x, action["native_increment"])
            change = value[:, 1] - value[:, 2]
            deleted, _ = contacts(change, deletion["raw"])
            arrays[f"{law}_{rule}_native"] = native
            arrays[f"{law}_{rule}_native_increment"] = increment
            arrays[f"{law}_{rule}_deletion"] = deleted
            arrays[f"{law}_{rule}_deletion_scores"] = change
            for mode, account in (
                ("native", native),
                ("native_increment", increment),
                ("deletion", deleted),
            ):
                for t, reader in np.ndindex(5, 5):
                    direct_rows.append(
                        {
                            "law": law,
                            "rule": rule,
                            "action": mode,
                            "meaning": cfg["targets"][t],
                            "reader": cfg["targets"][reader],
                            "contact": clean(account[t, reader]),
                        }
                    )
            geometry = (
                B.gram(value[:, 0] - value[:, 0].mean(0))
                if np.isfinite(value[:, 0]).all()
                else None
            )
            composition = value[:, 2] - value[:, 1] - 0.5 * (value[:, 3] - value[:, 4])
            arrays[f"{law}_{rule}_composition"] = composition
            connection_summary[law][rule] = {
                "native_diagonal": clean(np.diag(native)),
                "increment_diagonal": clean(np.diag(increment)),
                "deletion_diagonal": clean(np.diag(deleted)),
                "discrimination_diagonal": clean(
                    [
                        arrays[f"{law}_{rule}_source_discrimination"][t, t, t]
                        for t in range(5)
                    ]
                ),
                "author_target_cosine": geometry["cosine"].tolist()
                if geometry
                else None,
                "probability_mean_composition_RMS": float(
                    np.sqrt(np.mean(composition**2))
                )
                if np.isfinite(composition).all()
                else None,
                "posterior_support_failures": int(
                    (~cold[law + "_supported"][:, :, r - 1]).sum()
                )
                if r
                else 0,
            }
    elapsed = time.monotonic() - started
    if elapsed > cfg["max_compute_seconds"]:
        raise RuntimeError("Finite CPU cap exceeded;do not automatically repeat")
    R.archive(out / "analysis.npz", **arrays)
    R.table(out / "source_channel_accounts.csv", source_rows)
    R.table(out / "meaning_action_tensor.csv", rows)
    R.table(out / "direct_contacts.csv", direct_rows)
    R.write(out / "target_grams.json", target_grams)
    R.write(
        out / "summary.json",
        {
            "connections": connection_summary,
            "source_channels": {
                name: {
                    "supported_cells": int(c["support"].sum()),
                    "centered_spectrum": c["centered_spectrum"].tolist(),
                }
                for name, c in channels.items()
            },
            "refinement_min_eigenvalue": float(gain_eig[0]),
            "garbling_error": float(np.max(abs(q @ garbling - p))),
            "source_targets": source_rows,
            "signed_tensor_rows": len(rows),
            "direct_rows": len(direct_rows),
            "analysis_CPU_seconds": elapsed,
            "not_accuracy_reliability_psychological_variance_or_novel_operator_theorem": True,
            "one_old_permutation_not_pvalue": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "source_channel_accounts.csv",
            "meaning_action_tensor.csv",
            "direct_contacts.csv",
            "target_grams.json",
            "summary.json",
        ),
    )
    print(
        json.dumps({"connections": connection_summary["actual"], "seconds": elapsed}),
        flush=True,
    )


def verify(out):
    """Replay all author histories by literal products and all2250 contacts."""
    cfg = check(out, ("cold_seal.json", "analyzed_seal.json"))["config"]
    z = S.selected(out / "observations.npz", ("cells", "masks", "edges"))
    bundles, bindings = (
        R.read(out / "source_bundles.json"),
        R.read(out / "bindings.json"),
    )
    with np.load(out / "cold_scores.npz", allow_pickle=False) as f:
        cold = {k: f[k] for k in f.files}
    with np.load(out / "analysis.npz", allow_pickle=False) as f:
        account = {k: f[k] for k in f.files}
    old = S.selected(
        ROOT / cfg["meaning_bridge_parent"] / "cold_scores.npz",
        ("actual_full_score", "permuted_full_score"),
    )
    action = S.selected(
        ROOT / cfg["action_parent"] / "analysis.npz", ("matrix", "interaction")
    )
    modes = {
        "total": action["matrix"],
        "source_discrimination": action["matrix"]
        - action["matrix"].mean(2, keepdims=True),
        "reader_source_interaction": action["interaction"],
    }
    history_error = mean_error = operator_error = contact_error = 0.0
    entries, histories = 0, 0
    for law, bundle in bundles.items():
        p, y = (
            np.asarray(bundle["source_mass"]),
            np.asarray(bundle["source_binary_targets"]),
        )
        pi, psi = np.asarray(bundle["prior"]), np.asarray(bundle["event_probability"])
        for u, binding in enumerate(bindings):
            for group, (cells, _) in enumerate(
                M.history_groups(z["cells"][u], z["masks"][u], binding["event_ids"])
            ):
                likelihood = np.prod(p[:, cells], axis=1)
                shared = (
                    likelihood @ y / likelihood.sum()
                    if likelihood.sum()
                    else np.full(5, np.nan)
                )
                l0 = np.prod(np.asarray(bundle["rho0"])[cells], axis=0) * (1 - pi)
                l1 = np.prod(np.asarray(bundle["rho1"])[cells], axis=0) * pi
                iid = np.divide(
                    l1, l0 + l1, out=np.full(5, np.nan), where=(l0 + l1) > 0
                )
                expected = np.stack([psi[cells].mean(0), iid, shared])
                observed = cold[law + "_probabilities"][u, group]
                if not np.array_equal(np.isfinite(expected), np.isfinite(observed)):
                    raise ValueError(
                        "Posterior support masks differ in alternate replay"
                    )
                history_error = max(
                    history_error,
                    float(
                        np.max(
                            abs(
                                expected[np.isfinite(expected)]
                                - observed[np.isfinite(expected)]
                            )
                        )
                    ),
                )
                histories += 1
        mean_error = max(
            mean_error,
            float(
                np.max(
                    abs(
                        cold[law + "_probabilities"][:, :3, 0]
                        - pi
                        - old[law + "_full_score"]
                    )
                )
            ),
        )
        for name in cfg["channels"]:
            mass = p if name == "single_event" else account["pair_mass"]
            ref = mass.mean(0)
            b = sum(
                np.outer(mass[:, a], mass[:, a]) / ref[a]
                for a in np.flatnonzero(ref > 0)
            ) / len(p)
            operator_error = max(
                operator_error, float(np.max(abs(b - account[name + "_round_trip"])))
            )
            g = y - pi
            operator_error = max(
                operator_error,
                float(np.max(abs(b @ g - account[f"{law}_{name}_return_scores"]))),
            )
        for rule in cfg["rules"]:
            x = account[f"{law}_{rule}_edge_scores"]
            for mode, values in modes.items():
                k = account[f"{law}_{rule}_{mode}"]
                for t, reader, encoding in np.ndindex(5, 5, 5):
                    keep = np.flatnonzero(np.isfinite(x[:, t]))
                    literal = (
                        sum(float(x[e, t] * values[e, reader, encoding]) for e in keep)
                        / len(keep)
                        if len(keep)
                        else np.nan
                    )
                    if np.isfinite(literal) != np.isfinite(k[t, reader, encoding]):
                        raise ValueError("Missing contact was zero-filled")
                    if np.isfinite(literal):
                        contact_error = max(
                            contact_error, abs(literal - k[t, reader, encoding])
                        )
                    entries += 1
    error = max(history_error, mean_error, operator_error, contact_error)
    if error > 2e-12:
        raise ValueError("Alternate accounting failed;no tuning or science rerun")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "history_product_error": history_error,
            "inherited_mean_error": mean_error,
            "literal_operator_error": operator_error,
            "contact_error": contact_error,
            "histories_verified": histories,
            "tensor_entries_verified": entries,
            "same_agent_alternate_arithmetic_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("HM234 verified", error, histories, entries, flush=True)


def main():
    """Run one finite immutable stage; never retry a failed scientific run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "score", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        check(
            args.out, ("cold_seal.json", "analyzed_seal.json", "verification_seal.json")
        )
        print("HM234 seals PASS;original18/100/44 scope OPEN")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
